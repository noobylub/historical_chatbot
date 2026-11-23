"""Deploy with: uv run --with modal modal deploy modal_historicalChat.py

The endpoint is public: anyone with its URL can incur GPU charges.
Model license: CC BY-NC-SA 4.0 (noncommercial use with attribution).
Model: https://huggingface.co/tventurella/Mr.Chatterbox-2.7B-Chat
"""

import subprocess

import modal

MODEL = "tventurella/Mr.Chatterbox-2.7B-Chat"
PORT = 8000

app = modal.App("chatterbox-chat")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers>=5.18.0,<6")
)
cache = modal.Volume.from_name("chatterbox-huggingface-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": cache},
    max_containers=1,
    scaledown_window=120,
    timeout=900,
)
@modal.web_server(PORT, startup_timeout=900, requires_proxy_auth=False)
def serve() -> None:
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            
        ]
    )