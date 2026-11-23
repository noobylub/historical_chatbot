"""Deploy with: modal deploy historical_chat/qwen/MODAL_LORA.py

Set LORA_ADAPTER to a run ID from the qwen35-9b-lora-training Modal Volume.
Requests select it using model="qwen35-9b-william-morris".
The endpoint is public and does not require a Modal proxy token.
Uses vLLM 0.34.0 and its latest compatible Transformers release, 5.22.2.
"""

import subprocess
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LORA_ADAPTER = Path("/adapters/qwen35-9b-william-morris-2edd3")
print(LORA_ADAPTER)
LORA_NAME = "qwen35-9b-william-morris"
PORT = 8000

app = modal.App("qwen35-9b-lora-serving")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.34.0", "transformers==5.22.2")
)
hf_cache = modal.Volume.from_name("qwen35-9b-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen35-9b-vllm-cache", create_if_missing=True)
adapter_volume = modal.Volume.from_name("qwen35-9b-lora-training")


def build_server_command(adapter):
    for filename in (
        "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja",
    ):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(f"Missing adapter file: {adapter / filename}")
    return [
        "vllm", "serve", MODEL,
        "--revision", MODEL_REVISION,
        "--tokenizer", str(adapter),
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--language-model-only",
        "--reasoning-parser", "qwen3",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--enable-lora",
        "--max-lora-rank", "16",
        "--lora-modules", f"{LORA_NAME}={adapter}",
        "--host", "0.0.0.0",
        "--port", str(PORT),
    ]


@app.server(
    image=image,
    gpu="L40S",
    port=PORT,
    unauthenticated=True,
    startup_timeout=1200,
    scaledown_window=300,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/adapters": adapter_volume.with_mount_options(read_only=True),
    },
)
class Server:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen(build_server_command(LORA_ADAPTER))

    @modal.exit()
    def stop(self):
        self.process.terminate()