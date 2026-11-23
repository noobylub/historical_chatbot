import argparse
import json
import os
import random
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def main():
    # Parses command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-five", action="store_true", help="Analyze five randomly selected texts")
    args = parser.parse_args()
    
     # Load environment variables and API key
    load_dotenv()
    api_key = os.getenv("open_router_key")
    if not api_key:
        raise ValueError("Missing open_router_key in the environment or .env file")


    # Read through the texts in the folder
    base_dir = Path(__file__).resolve().parent
    texts_dir = base_dir.parent / "morris_corpus_labeled" / "texts"
    output_file = base_dir / ("lora_results_random_five.json" if args.random_five else "lora_results.json")
    if not texts_dir.is_dir():
        raise FileNotFoundError(f"Text directory not found: {texts_dir}")
    text_files = sorted(texts_dir.glob("*.txt"))
    if not text_files:
        raise ValueError(f"No text files found in {texts_dir}")
    if args.random_five:
        if len(text_files) < 5:
            raise ValueError(f"Need at least five text files in {texts_dir}")
        text_files = random.sample(text_files, 5)

   
    # Load the LoRA instructions and map each text file to its source title.
    prompt = json.loads((base_dir / "prompts.json").read_text(encoding="utf-8"))["LORA_instructions"]
    instructions, input_template = prompt.split("## INPUT", 1)
    manifest = json.loads((texts_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    titles = {}
    for record in manifest:
        if record.get("text_file"):
            titles[record["text_file"]] = record["title"]
        for part in record.get("parts", []):
            if part.get("text_file"):
                titles[part["text_file"]] = record["title"]
    
    # Initialize the results dictionary to store analysis outcomes for each text file
    results = {}
    if not args.random_five and output_file.exists():
        results = json.loads(output_file.read_text(encoding="utf-8"))
        if not isinstance(results, dict):
            raise ValueError(f"Expected a JSON object in {output_file}")
    failed = []
    for text_file in text_files:
        if text_file.name in results:
            continue
        print(f"Analyzing {text_file.name}", flush=True)
        text = text_file.read_text(encoding="utf-8")
        title = titles[f"texts/{text_file.name}"]
        user_input = input_template.replace("{source_title}", title).replace("{source_text}", text)
        for attempt in range(1, 4):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "openai/gpt-6-luna",
                    "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": user_input}],
                    "reasoning": {"enabled": False},
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            try:
                if not isinstance(content, str):
                    raise ValueError("Response content is not text")
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")
            except ValueError as exc:
                print(f"{text_file.name}: invalid response on attempt {attempt}/3: {exc}", file=sys.stderr)
                continue
            break
        else:
            failed.append(text_file.name)
            continue
        results[text_file.name] = result
        output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total results: {len(results)}")
    if failed:
        raise SystemExit(f"Could not analyze {len(failed)} text(s): {', '.join(failed)}")


if __name__ == "__main__":
    main()