import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


SCORE_NAMES = {
    "historical_faithfulness",
    "language_authenticity",
    "conversational_quality",
    "persona_training_value",
}
ISSUE_LABELS = {
    "unsupported_claim",
    "incorrect_attribution",
    "fictional_character_voice",
    "language_distortion",
    "poor_question_alignment",
    "incoherent_answer",
    "insufficient_context",
    "low_persona_relevance",
    "uncertain_authorship",
    "other",
}


class InvalidEvaluationError(ValueError):
    pass


def validate_evaluations(evaluations, examples, filename):
    if not isinstance(evaluations, list) or len(evaluations) != len(examples):
        raise ValueError(f"Expected {len(examples)} evaluations")

    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            raise ValueError(f"Evaluation {index} is not an object")
        if type(evaluation.get("index")) is not int or evaluation["index"] != index:
            raise ValueError(f"Evaluation {index} has an incorrect index")
        if evaluation.get("filename") != filename:
            raise ValueError(f"Evaluation {index} has an incorrect filename")
        for field in ("question", "answer"):
            if evaluation.get(field) != examples[index][field]:
                raise ValueError(f"Evaluation {index} has an incorrect {field}")
        if any(field in evaluation for field in ("original_excerpt", "modifications", "grounded")):
            raise ValueError(f"Evaluation {index} contains input-only fields")
        decision = evaluation.get("decision")
        if decision not in ("KEEP", "REVISE", "OMIT"):
            raise ValueError(f"Evaluation {index} has an invalid decision")
        if evaluation.get("include_in_training") is not (decision == "KEEP"):
            raise ValueError(f"Evaluation {index} has an inconsistent training flag")
        scores = evaluation.get("scores")
        if not isinstance(scores, dict) or set(scores) != SCORE_NAMES:
            raise ValueError(f"Evaluation {index} has invalid score categories")
        if any(type(score) is not int or not 1 <= score <= 5 for score in scores.values()):
            raise ValueError(f"Evaluation {index} has an invalid score")
        if not isinstance(evaluation.get("reason"), str) or not evaluation["reason"].strip():
            raise ValueError(f"Evaluation {index} has no reason")
        issues = evaluation.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or issue not in ISSUE_LABELS for issue in issues
        ):
            raise ValueError(f"Evaluation {index} has invalid issue labels")


def evaluate_examples(filename, examples, prompt, api_key):
    for attempt in range(1, 6):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "openai/gpt-6-luna",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(
                        {"filename": filename, "examples": examples}, ensure_ascii=False
                    )},
                ],
                "reasoning": {"enabled": False},
            },
            timeout=120,
        )
        if response.status_code == 400:
            print(f"{filename}: HTTP 400 on attempt {attempt}/5: {response.text}", file=sys.stderr)
            continue
        if not response.ok:
            print(response.text, file=sys.stderr)
        response.raise_for_status()
        try:
            message = response.json()["choices"][0]["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("Response content is not text")
            evaluations = json.loads(content)
            validate_evaluations(evaluations, examples, filename)
            return evaluations
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            print(f"{filename}: invalid response on attempt {attempt}/5: {exc}", file=sys.stderr)
    raise InvalidEvaluationError("Could not obtain a valid evaluation after five attempts")


def save_results(output_file, results):
    temporary_file = output_file.with_suffix(".json.tmp")
    temporary_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_file.replace(output_file)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-five", action="store_true",
        help="Evaluate only the first five examples across all sources",
    )
    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    sources = json.loads((base_dir / "lora_results.json").read_text(encoding="utf-8"))
    prompt = json.loads((base_dir / "prompts.json").read_text(encoding="utf-8"))["evaluation"]
    output_file = base_dir / (
        "instructions_eval_results_first_five.json" if args.first_five else "instructions_eval_results.json"
    )
    if not isinstance(sources, dict) or not isinstance(prompt, str):
        raise ValueError("Expected a source object and an evaluation prompt")
    results = json.loads(output_file.read_text(encoding="utf-8")) if output_file.exists() else {}
    if not isinstance(results, dict):
        raise ValueError(f"Expected a JSON object in {output_file}")

    load_dotenv()
    api_key = os.getenv("open_router_key")
    if not api_key:
        raise ValueError("Missing open_router_key in the environment or .env file")

    remaining = 5 if args.first_five else None
    evaluated_count = 0
    skipped = []
    for filename, source in sources.items():
        if remaining == 0:
            break
        if not isinstance(source, dict) or not isinstance(source.get("examples"), list):
            raise ValueError(f"{filename}: expected an examples array")
        examples = source["examples"]
        if remaining is not None:
            examples = examples[:remaining]
            remaining -= len(examples)
            if not examples:
                continue
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or any(
                not isinstance(example.get(field), str) for field in ("question", "answer")
            ):
                raise ValueError(f"{filename}: example {index} must have a question and answer")
        if filename in results:
            evaluations = results[filename]
            migrated = False
            if isinstance(evaluations, list) and len(evaluations) == len(examples):
                for evaluation, example in zip(evaluations, examples):
                    if isinstance(evaluation, dict) and not any(
                        field in evaluation for field in ("filename", "question", "answer")
                    ):
                        evaluation.update(filename=filename, question=example["question"], answer=example["answer"])
                        migrated = True
            validate_evaluations(evaluations, examples, filename)
            if migrated:
                save_results(output_file, results)
            evaluated_count += len(examples)
            continue
        print(f"Evaluating {filename}: {len(examples)} examples", flush=True)
        try:
            results[filename] = evaluate_examples(filename, examples, prompt, api_key) if examples else []
        except InvalidEvaluationError as exc:
            print(f"Skipping {filename}: {exc}", file=sys.stderr)
            skipped.append(filename)
            continue
        save_results(output_file, results)
        evaluated_count += len(examples)

    print(f"Evaluated {evaluated_count} examples")
    if skipped:
        print(f"Skipped {len(skipped)} source(s): {', '.join(skipped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
