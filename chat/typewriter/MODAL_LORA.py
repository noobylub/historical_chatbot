"""Deploy with: modal deploy historical_chat/MODAL_LORA.py

The trained adapter is read from the typewriter-lora-training Modal Volume.
Requests select it using model="william-morris".
The endpoint is public and does not require a Modal proxy token.
Uses vLLM 0.34.0 and its latest compatible Transformers release, 5.22.2.
"""

import subprocess
from pathlib import Path

import modal

MODEL = "typewriter-ai/typewriter-1913-v2-selfinstruct"
MODEL_REVISION = "55ef3658a8d3c6ec688e2e24c9c378f3d8a97637"
LORA_ADAPTER = Path("/adapters/william-morris-4-epoch-ace48")
LORA_NAME = "william-morris"
PORT = 8000

app = modal.App("typewriter-lora")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.34.0", "transformers==5.22.2")
)
hf_cache = modal.Volume.from_name("typewriter-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("typewriter-vllm-cache", create_if_missing=True)
adapter_volume = modal.Volume.from_name("typewriter-lora-training")


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
    # Mounting the necessary volumes for caching and adapters
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