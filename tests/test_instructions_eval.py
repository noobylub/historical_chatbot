import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from WM_analysis import instructions_eval


def example(question):
    return {
        "question": question,
        "answer": "The answer.",
        "original_excerpt": "Evidence.",
        "modifications": [],
        "grounded": True,
    }


def evaluation(index=0, filename="first.txt", question="First?"):
    return {
        "filename": filename,
        "index": index,
        "question": question,
        "answer": "The answer.",
        "decision": "KEEP",
        "include_in_training": True,
        "scores": {
            "historical_faithfulness": 5,
            "language_authenticity": 5,
            "conversational_quality": 5,
            "persona_training_value": 5,
        },
        "reason": "The answer is supported.",
        "issues": [],
    }


def response(content):
    result = Mock()
    result.json.return_value = {"choices": [{"message": {"content": content}}]}
    return result


class InstructionsEvalTests(unittest.TestCase):
    @patch("WM_analysis.instructions_eval.requests.post")
    def test_sends_array_and_retries_invalid_results(self, post):
        examples = [example("First?"), example("Second?")]
        post.side_effect = [
            response(json.dumps([evaluation()])),
            response(json.dumps([evaluation(), evaluation(1, question="Second?")])),
        ]

        result = instructions_eval.evaluate_examples("first.txt", examples, "evaluation prompt", "test-key")

        self.assertEqual(result, [evaluation(), evaluation(1, question="Second?")])
        self.assertEqual(post.call_count, 2)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "openai/gpt-6-luna")
        self.assertEqual(payload["messages"][0]["content"], "evaluation prompt")
        self.assertEqual(json.loads(payload["messages"][1]["content"]), {
            "filename": "first.txt", "examples": examples
        })
        self.assertNotIn("response_format", payload)

    def test_rejects_invalid_decisions_and_scores(self):
        bad_flag = evaluation()
        bad_flag["include_in_training"] = False
        with self.assertRaisesRegex(ValueError, "training flag"):
            instructions_eval.validate_evaluations([bad_flag], [example("First?")], "first.txt")

        bad_score = evaluation()
        bad_score["scores"]["language_authenticity"] = True
        with self.assertRaisesRegex(ValueError, "invalid score"):
            instructions_eval.validate_evaluations([bad_score], [example("First?")], "first.txt")

    def test_rejects_modified_identity_or_input_only_fields(self):
        examples = [example("First?")]
        for field, value in (("filename", "wrong.txt"), ("question", "Changed?"),
                             ("answer", "Changed."), ("original_excerpt", "Evidence.")):
            with self.subTest(field=field):
                result = evaluation()
                result[field] = value
                with self.assertRaises(ValueError):
                    instructions_eval.validate_evaluations([result], examples, "first.txt")

    @patch("WM_analysis.instructions_eval.requests.post")
    def test_accepts_valid_response_on_fifth_attempt(self, post):
        post.side_effect = [response("invalid")] * 4 + [response(json.dumps([evaluation()]))]
        with redirect_stderr(io.StringIO()):
            result = instructions_eval.evaluate_examples("first.txt", [example("First?")], "prompt", "test-key")
        self.assertEqual(post.call_count, 5)
        self.assertEqual(result, [evaluation()])

    @patch("WM_analysis.instructions_eval.requests.post")
    def test_retries_http_400_and_accepts_valid_response(self, post):
        bad_response = Mock(status_code=400, text="Bad request")
        post.side_effect = [bad_response, response(json.dumps([evaluation()]))]
        errors = io.StringIO()
        with redirect_stderr(errors):
            result = instructions_eval.evaluate_examples("first.txt", [example("First?")], "prompt", "test-key")
        self.assertEqual(result, [evaluation()])
        self.assertEqual(post.call_count, 2)
        self.assertIn("first.txt: HTTP 400 on attempt 1/5: Bad request", errors.getvalue())

    @patch("WM_analysis.instructions_eval.requests.post")
    def test_fails_after_five_http_400_responses(self, post):
        post.return_value = Mock(status_code=400, text="Bad request")
        errors = io.StringIO()
        with redirect_stderr(errors):
            with self.assertRaises(instructions_eval.InvalidEvaluationError):
                instructions_eval.evaluate_examples("first.txt", [example("First?")], "prompt", "test-key")
        self.assertEqual(post.call_count, 5)
        self.assertIn("first.txt: HTTP 400 on attempt 5/5: Bad request", errors.getvalue())

    @patch("WM_analysis.instructions_eval.load_dotenv")
    @patch("WM_analysis.instructions_eval.requests.post")
    def test_persists_results_and_resumes_without_repeating_calls(self, post, _load_dotenv):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources = {
                "first.txt": {"examples": [example("First?")]},
                "empty.txt": {"examples": []},
                "second.txt": {"examples": [example("Second?")]},
            }
            (base / "lora_results.json").write_text(json.dumps(sources), encoding="utf-8")
            (base / "prompts.json").write_text(json.dumps({"evaluation": "prompt"}), encoding="utf-8")
            post.side_effect = [response(json.dumps([evaluation()])), RuntimeError("interrupted")]

            with patch.object(instructions_eval, "__file__", str(base / "instructions_eval.py")), \
                 patch.dict("os.environ", {"open_router_key": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    instructions_eval.main([])
                output = base / "instructions_eval_results.json"
                self.assertEqual(json.loads(output.read_text()), {
                    "first.txt": [evaluation()], "empty.txt": []
                })

                post.side_effect = [response(json.dumps([evaluation(filename="second.txt", question="Second?")]))]
                post.reset_mock()
                instructions_eval.main([])
                self.assertEqual(post.call_count, 1)
                self.assertEqual(json.loads(output.read_text()), {
                    "first.txt": [evaluation()], "empty.txt": [],
                    "second.txt": [evaluation(filename="second.txt", question="Second?")]
                })

    @patch("WM_analysis.instructions_eval.load_dotenv")
    @patch("WM_analysis.instructions_eval.requests.post")
    def test_first_five_spans_sources_and_uses_separate_results(self, post, _load_dotenv):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources = {
                "first.txt": {"examples": [example(f"First {i}?") for i in range(3)]},
                "empty.txt": {"examples": []},
                "second.txt": {"examples": [example(f"Second {i}?") for i in range(4)]},
                "third.txt": {"examples": [example("Not selected?")]},
            }
            (base / "lora_results.json").write_text(json.dumps(sources), encoding="utf-8")
            (base / "prompts.json").write_text(json.dumps({"evaluation": "prompt"}), encoding="utf-8")
            full_output = base / "instructions_eval_results.json"
            full_output.write_text('{"untouched": []}', encoding="utf-8")
            post.side_effect = [
                response(json.dumps([evaluation(i, question=f"First {i}?") for i in range(3)])),
                response(json.dumps([evaluation(i, filename="second.txt", question=f"Second {i}?") for i in range(2)])),
            ]

            with patch.object(instructions_eval, "__file__", str(base / "instructions_eval.py")), \
                 patch.dict("os.environ", {"open_router_key": "test-key"}):
                instructions_eval.main(["--first-five"])
                self.assertEqual(post.call_count, 2)
                sent = [json.loads(call.kwargs["json"]["messages"][1]["content"]) for call in post.call_args_list]
                self.assertEqual(sent, [
                    {"filename": "first.txt", "examples": sources["first.txt"]["examples"]},
                    {"filename": "second.txt", "examples": sources["second.txt"]["examples"][:2]},
                ])
                output = base / "instructions_eval_results_first_five.json"
                self.assertEqual(json.loads(output.read_text()), {
                    "first.txt": [evaluation(i, question=f"First {i}?") for i in range(3)],
                    "second.txt": [evaluation(i, filename="second.txt", question=f"Second {i}?") for i in range(2)],
                })
                self.assertEqual(json.loads(full_output.read_text()), {"untouched": []})
                post.reset_mock()
                instructions_eval.main(["--first-five"])
                post.assert_not_called()

    @patch("WM_analysis.instructions_eval.load_dotenv")
    @patch("WM_analysis.instructions_eval.requests.post")
    def test_migrates_existing_results_without_re_evaluating(self, post, _load_dotenv):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "lora_results.json").write_text(
                json.dumps({"first.txt": {"examples": [example("First?")]}}), encoding="utf-8"
            )
            (base / "prompts.json").write_text(json.dumps({"evaluation": "prompt"}), encoding="utf-8")
            output = base / "instructions_eval_results_first_five.json"
            legacy = evaluation()
            for field in ("filename", "question", "answer"):
                del legacy[field]
            output.write_text(json.dumps({"first.txt": [legacy]}), encoding="utf-8")
            with patch.object(instructions_eval, "__file__", str(base / "instructions_eval.py")), \
                 patch.dict("os.environ", {"open_router_key": "test-key"}):
                instructions_eval.main(["--first-five"])
            post.assert_not_called()
            self.assertEqual(json.loads(output.read_text()), {"first.txt": [evaluation()]})

    @patch("WM_analysis.instructions_eval.load_dotenv")
    @patch("WM_analysis.instructions_eval.requests.post")
    def test_skips_source_after_five_invalid_responses_and_retries_on_next_run(self, post, _load_dotenv):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources = {
                "first.txt": {"examples": [example("First?")]},
                "second.txt": {"examples": [example("Second?")]},
            }
            (base / "lora_results.json").write_text(json.dumps(sources), encoding="utf-8")
            (base / "prompts.json").write_text(json.dumps({"evaluation": "prompt"}), encoding="utf-8")
            post.side_effect = [response("invalid")] * 5 + [
                response(json.dumps([evaluation(filename="second.txt", question="Second?")]))
            ]
            errors, output_text = io.StringIO(), io.StringIO()
            with patch.object(instructions_eval, "__file__", str(base / "instructions_eval.py")), \
                 patch.dict("os.environ", {"open_router_key": "test-key"}), \
                 redirect_stderr(errors), redirect_stdout(output_text):
                instructions_eval.main([])
                self.assertEqual(post.call_count, 6)
                self.assertIn("attempt 5/5", errors.getvalue())
                self.assertIn("Skipped 1 source(s): first.txt", errors.getvalue())
                self.assertIn("Evaluated 1 examples", output_text.getvalue())
                output = base / "instructions_eval_results.json"
                self.assertEqual(json.loads(output.read_text()), {
                    "second.txt": [evaluation(filename="second.txt", question="Second?")]
                })
                post.side_effect = [response(json.dumps([evaluation()]))]
                post.reset_mock()
                instructions_eval.main([])
                self.assertEqual(post.call_count, 1)
                self.assertEqual(len(json.loads(output.read_text())), 2)


if __name__ == "__main__":
    unittest.main()
