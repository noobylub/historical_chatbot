# Train Typewriter LoRA

Authenticate with Modal and ensure your account can use an A100 80 GB GPU.
From the project root, run:

```sh
.venv/bin/modal run historical_chat/TRAIN_LORA.py
```

Optionally choose a run name:

```sh
.venv/bin/modal run historical_chat/TRAIN_LORA.py --run-name william-morris-v2
```

The folder name includes a five-character random hexadecimal suffix, for example
`william-morris-v2-a3f91`. Names must start with a letter or number and contain
only letters, numbers, underscores, and hyphens (up to 200 characters). The
default name is `william-morris`. A collision with an existing Volume folder
fails the run rather than overwriting it.

Keep the command running until the adapter downloads to
`lora_adapters/<run_id>/`. GPU compute and Modal Volume storage incur charges.
The one-shot GPU container shuts down after the job; the Volume persists until
you remove its contents. If the download fails, the command prints a retry
command. A failed reload check also reports the saved Volume path.

The script uses the pinned model revision, 16-bit LoRA, two epochs, and a
source-grouped validation split. It logs loss and saves metrics in
`training_info.json`. The training sequence limit is 2,048 tokens; longer
answers are truncated and counted in that file. The base model supports up to
8,192 tokens.

Each example starts with this system message:

> You are William Morris, an English textile designer, poet, artist, writer, and socialist activist associated with the British Arts and Crafts movement.

Use the same system message when prompting the trained adapter.

## Load the adapter

The adapter is not a standalone model. With PyTorch, PEFT, and Transformers
installed, load it alongside the same base model and revision:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = "lora_adapters/<run_id>"
base = "typewriter-ai/typewriter-1913-v2-selfinstruct"
revision = "55ef3658a8d3c6ec688e2e24c9c378f3d8a97637"
tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(
    base, revision=revision, dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, adapter).eval()
```

The 7B base model requires substantial memory. Use the saved tokenizer's chat
template for prompts. It includes a beginning-of-sequence token, so do not add
another one when tokenizing the rendered prompt.

## Serve the trained adapter

The Modal serving script loads run `264afb02c10e4577a6d53275f0d1d68c` from the
`typewriter-lora-training` Volume, using that run's saved tokenizer and chat
template. Deploy with:

```sh
modal deploy historical_chat/MODAL_LORA.py
```

In chat-completion requests, set `"model": "william-morris"` and include the
system message above. Requests using the base model name do not use the adapter.
The endpoint is public: anyone with its URL can incur GPU charges. To serve a
different run, update `LORA_ADAPTER` in the serving script to its Volume path.
