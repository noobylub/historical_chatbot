"""Train Typewriter LoRA on Modal and download the adapter.

Run from the project root: .venv/bin/modal run historical_chat/TRAIN_LORA.py
Keep the command running until the download completes. Modal authentication and
a GPU-enabled account are required. GPU time and persistent Modal Volume
storage incur charges.
Adapters are downloaded to lora_adapters/<run_id>/ and require the base model
and revision recorded in training_info.json. If download fails, retry the printed
Modal Volume command. Delete Volume data separately when no longer needed.
"""

import json
import math
import random
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import modal

# Pin the base model and tokenizer to the same revision used for inference.
MODEL = "typewriter-ai/typewriter-1913-v2-selfinstruct"
MODEL_REVISION = "55ef3658a8d3c6ec688e2e24c9c378f3d8a97637"
# Include this same system message when prompting the trained adapter.
SYSTEM_PROMPT = (
    "You are William Morris, an English textile designer, poet, artist, writer, "
    "and socialist activist associated with the British Arts and Crafts movement."
)
DATA_PATH = Path(__file__).resolve().parents[1] / "WM_analysis/instructions_eval_results.json"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "lora_adapters"
VOLUME_NAME = "typewriter-lora-training"
MAX_LENGTH = 2048
SEED = 3407
# Adapt all attention and MLP projection matrices in the Llama architecture.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# The image is built in Modal; these GPU dependencies are not needed locally.
app = modal.App("typewriter-lora-training")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "unsloth[huggingface]==2026.11.9", "bitsandbytes", "transformers==5.17.0",
    )
)
# A persistent Volume keeps the adapter available even after the GPU job exits.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def load_examples(path):
    """Keep approved question-answer pairs and split them by source document."""
    with path.open(encoding="utf-8") as file:
        groups = json.load(file)
    if not isinstance(groups, dict):
        raise ValueError("Expected an object mapping source filenames to lists.")

    kept = {}
    for source, rows in groups.items():
        if not isinstance(rows, list):
            raise ValueError(f"{source}: expected a list of evaluations.")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{source}: expected an evaluation object.")
            if row.get("decision") != "KEEP":
                continue
            question, answer = row.get("question"), row.get("answer")
            if not all(isinstance(value, str) and value.strip() for value in (question, answer)):
                raise ValueError(f"{source}: KEEP example has an empty question or answer.")
            # Evaluation scores and other metadata are not passed to training.
            kept.setdefault(source, []).append({"question": question, "answer": answer})

    if len(kept) < 2:
        raise ValueError("Need KEEP examples from at least two sources for validation.")
    # Hold out whole source documents to reduce overlap between train and
    # validation; sorting and the fixed seed make the split reproducible.
    sources = sorted(kept)
    random.Random(SEED).shuffle(sources)
    validation_sources = set(sources[:max(1, round(len(sources) * 0.05))])
    train = [row for source in sources if source not in validation_sources for row in kept[source]]
    validation = [row for source in sources if source in validation_sources for row in kept[source]]
    return train, validation


def tokenize_examples(examples, tokenizer):
    """Apply Typewriter's chat template and create labels for assistant answers."""
    tokenized = []
    truncated = 0
    for example in examples:
        question = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        conversation = question + [{"role": "assistant", "content": example["answer"]}]
        # The generation prompt ends with the assistant role marker. It must
        # match the start of the complete conversation so the loss mask aligns.
        prefix = tokenizer.apply_chat_template(
            question, tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        tokens = tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False, return_dict=True
        )["input_ids"]
        empty_answer = tokenizer.apply_chat_template(
            question + [{"role": "assistant", "content": ""}],
            tokenize=True, add_generation_prompt=False, return_dict=True,
        )["input_ids"]
        if tokens[:len(prefix)] != prefix or empty_answer[:len(prefix)] != prefix:
            raise ValueError("Chat template did not produce the expected answer prefix.")
        # Typewriter ends an assistant message with EOS followed by a newline.
        suffix = empty_answer[len(prefix):]
        if not suffix or suffix[0] != tokenizer.eos_token_id or tokens[-len(suffix):] != suffix:
            raise ValueError("Chat template did not produce the expected answer ending.")
        if len(prefix) + len(suffix) >= MAX_LENGTH:
            raise ValueError("A question is too long to leave room for an answer and its ending.")
        if len(tokens) <= len(prefix) + len(suffix):
            raise ValueError("Chat template did not produce any answer tokens.")
        if len(tokens) > MAX_LENGTH:
            # Keep the prompt and complete message ending when truncating.
            tokens = tokens[:MAX_LENGTH - len(suffix)] + suffix
            truncated += 1
        # -100 tells the loss function to ignore the system and user prompt;
        # the answer and its EOS remain supervised.
        tokenized.append({"input_ids": tokens, "labels": [-100] * len(prefix) + tokens[len(prefix):]})
    return tokenized, truncated


# One GPU container handles one job; the execution timeout is 24 hours and the
# short scale-down window avoids deliberately keeping an idle GPU warm.
@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=86400,
    scaledown_window=2,
    single_use_containers=True,
    volumes={"/outputs": volume},
)
def train_adapter(train_examples, validation_examples, run_id):
    """Run training and persist the adapter inside the Modal GPU container."""
    output = Path("/outputs") / run_id
    if output.exists():
        raise FileExistsError(f"Adapter output already exists: {output}")

    # Unsloth must be imported before transformers and PEFT.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    # Load the original BF16 model, not a 4-bit quantized QLoRA base.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        revision=MODEL_REVISION,
        max_seq_length=MAX_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        use_exact_model_name=True,
    )
    train_rows, train_truncated = tokenize_examples(train_examples, tokenizer)
    validation_rows, validation_truncated = tokenize_examples(validation_examples, tokenizer)
    print(f"Training: {len(train_rows)} examples; validation: {len(validation_rows)} examples.")
    print(f"Truncated answers: train={train_truncated}, validation={validation_truncated}.")

    # Attach rank-16 adapters to the attention and MLP projections.
    # Gradient checkpointing trades additional computation for lower VRAM use.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
    )
    model.print_trainable_parameters()
    # Catch architecture or target-name mismatches before training, and ensure
    # that the frozen base model is not accidentally being fully fine-tuned.
    for target in TARGET_MODULES:
        count = sum(
            name.endswith(f".{target}") and hasattr(module, "lora_A")
            for name, module in model.named_modules()
        )
        if count != model.config.num_hidden_layers:
            raise RuntimeError(f"Expected LoRA on every {target} layer, found {count}.")
    if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
        raise RuntimeError("The base model unexpectedly has trainable parameters.")

    # Single-example, unpadded batches avoid masking padding when pad and EOS
    # share the same token.
    def collate(rows):
        if len(rows) != 1:
            raise ValueError("This collator requires a batch size of one.")
        return {key: torch.tensor([value], dtype=torch.long) for key, value in rows[0].items()}

    # Run a forward/backward smoke check before committing to a full epoch.
    # It checks for finite loss and at least one nonzero finite LoRA gradient,
    # but does not prove that the entire training run will succeed.
    model.train()
    batch = {key: value.to("cuda") for key, value in collate(train_rows[:1]).items()}
    loss = model(**batch).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("The Unsloth compatibility check produced an invalid loss.")
    loss.backward()
    gradients = [p.grad for name, p in model.named_parameters() if "lora_" in name and p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("The Unsloth compatibility check produced invalid LoRA gradients.")
    if not any(torch.count_nonzero(grad).item() for grad in gradients):
        raise RuntimeError("The Unsloth compatibility check produced only zero LoRA gradients.")
    # The smoke check is not a training step; discard its gradients.
    model.zero_grad(set_to_none=True)
    del batch, loss, gradients

    # Accumulate 16 single-example batches per optimizer step. Evaluate once
    # per epoch; intermediate Trainer checkpoints are not saved.
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/typewriter-training",
            num_train_epochs=4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=0.05,
            lr_scheduler_type="linear",
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            seed=SEED,
        ),
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        data_collator=collate,
    )
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Training finished with a non-finite loss; adapter was not saved.")
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]
    if not eval_losses or not all(math.isfinite(value) for value in eval_losses):
        raise RuntimeError("Validation did not produce a finite loss; adapter was not saved.")

    # Save the adapter and the tokenizer used for training, not a merged base
    # model. Keep the revision and metrics alongside them for later reuse.
    output.mkdir()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "training_info.json").write_text(json.dumps({
        "base_model": MODEL,
        "base_revision": MODEL_REVISION,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_truncated": train_truncated,
        "validation_truncated": validation_truncated,
        "training_loss": result.training_loss,
        "validation_loss": eval_losses[-1],
        "log_history": trainer.state.log_history,
    }, indent=2), encoding="utf-8")
    # Commit before the reload check so saved files can be recovered if that
    # check fails or the local download is interrupted.
    volume.commit()

    # Reload against the in-memory base model and check one validation loss.
    # This checks loadability, not the quality of generated responses.
    model.load_adapter(str(output), adapter_name="reload_check", is_trainable=False)
    model.set_adapter("reload_check")
    model.eval()
    batch = {key: value.to("cuda") for key, value in collate(validation_rows[:1]).items()}
    with torch.inference_mode():
        loss = model(**batch).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Adapter reload check failed; saved files remain at {VOLUME_NAME}/{run_id}.")
    return run_id


@app.local_entrypoint()
# We also set a name to the lora download
def main(run_name: str = "william-morris-4-epoch"):
    """Prepare data locally, wait for the remote job, then download its result."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", run_name):
        raise ValueError("Run name must be 1-200 letters, numbers, underscores, or hyphens, and start with a letter or number.")
    train, validation = load_examples(DATA_PATH)
    run_id = f"{run_name}-{uuid.uuid4().hex[:5]}"
    destination = OUTPUT_ROOT / run_id
    print(f"Starting training with {len(train)} train and {len(validation)} validation examples.")
    print(f"Adapter recovery path: Modal Volume {VOLUME_NAME}, directory {run_id}")
    # The call blocks until training and the reload check finish. A failed
    # remote job raises here and does not attempt a local download.
    train_adapter.remote(train, validation, run_id)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "modal", "volume", "get", VOLUME_NAME, run_id, str(OUTPUT_ROOT)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        # --force allows recovery from files left by a partial download.
        retry = command[:5] + ["--force"] + command[5:]
        print(f"Download failed. Retry with: {shlex.join(retry)}", file=sys.stderr)
        raise
    print(f"Adapter downloaded to {destination}")