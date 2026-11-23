"""One chat request: uv run python chat_history.py "Hello!" """

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODEL = "tventurella/Mr.Chatterbox-2.7B-Chat"
ENDPOINT = "https://noobylub--chatterbox-chat-serve.us-east.modal.run"


def chat(prompt: str, endpoint: str = ENDPOINT) -> str:
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.8,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=900) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one message to Chatterbox.")
    parser.add_argument("prompt", help="Message to send")
    parser.add_argument("--endpoint", default=ENDPOINT, help="Modal base endpoint URL")
    args = parser.parse_args()
    try:
        print(chat(args.prompt, endpoint=args.endpoint))
    except HTTPError as exc:
        raise SystemExit(
            f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
        ) from exc


if __name__ == "__main__":
    main()