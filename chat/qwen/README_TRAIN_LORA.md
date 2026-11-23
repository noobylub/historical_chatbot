# Train a Qwen3.5-9B LoRA

Authenticate with Modal and ensure your account can use an A100 80 GB GPU.
From the project root, run:

```sh
.venv/bin/modal run historical_chat/qwen/TRAIN_LORAQWEN.py
```

Optionally choose a run name:

```sh
.venv/bin/modal run historical_chat/qwen/TRAIN_LORAQWEN.py --run-name qwen35-9b-william-morris-v2
```

The folder name includes a five-character random hexadecimal suffix, for
example `qwen35-9b-william-morris-v2-a3f91`. Names must start with a letter or
number and contain only letters, numbers, underscores, and hyphens (up to 200
characters). The default name is `qwen35-9b-william-morris`. A collision with an
existing Volume folder fails rather than overwriting it.

Keep the command running until the adapter downloads to
`lora_adapters/qwen/<run_id>/`. Training uses the pinned
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16 LoRA, four epochs, and a
source-grouped validation split. GPU compute and Modal Volume storage incur
charges. If the download fails, the command prints a retry command. The Modal
Volume persists until you remove its contents.

Training is text-only with a 2,048-token sequence limit; the base model supports
up to 262,144 tokens and image/video input. Longer examples are truncated and
counted in `training_info.json`. The adapter uses the non-thinking Qwen chat
template and the following system message:

> You are William Morris, an English textile designer, poet, artist, writer, and socialist activist associated with the British Arts and Crafts movement.

Use the same system message when prompting the trained adapter.

## Load the adapter

The adapter is not a standalone model. Load it with the same base model and
revision using Transformers and PEFT:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

adapter = "lora_adapters/qwen/<run_id>"
base = "Qwen/Qwen3.5-9B"
revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForImageTextToText.from_pretrained(
    base, revision=revision, dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, adapter).eval()
```

The adapter is trained for text-only use. The tokenizer saved with the adapter
contains Qwen's chat template.

## Serve the trained adapter

After training, replace `LORA_ADAPTER` in
[`MODAL_LORA.py`](./MODAL_LORA.py) with the full run ID printed by the training
command. Then deploy from the project root:

```sh
modal deploy historical_chat/qwen/MODAL_LORA.py
```

In chat-completion requests, set `"model": "qwen35-9b-william-morris"` and
include the system message above. The server uses the same pinned base-model
revision, loads the saved tokenizer, and serves the language model without its
vision encoder. The endpoint is public: anyone with its URL can incur GPU
charges.
