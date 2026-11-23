import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from historical_chat.typewriter import TRAIN_LORA
from historical_chat.typewriter.TRAIN_LORA import MAX_LENGTH, load_examples, tokenize_examples


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        prefix = [1] + [10] * len(messages[0]["content"]) + [2]
        prefix += [10] * len(messages[1]["content"]) + [3]
        if add_generation_prompt:
            return {"input_ids": prefix}
        return {"input_ids": prefix + [20] * len(messages[2]["content"]) + [self.eos_token_id, 42]}


class TrainLoraTests(unittest.TestCase):
    def test_filters_and_splits_by_source_without_metadata(self):
        data = {
            "a": [
                {"decision": "KEEP", "question": "a1", "answer": "one", "reason": "extra"},
                {"decision": "KEEP", "question": "a2", "answer": "two"},
                {"decision": "REVISE", "question": "no", "answer": "no"},
            ],
            "b": [{"decision": "KEEP", "question": "b1", "answer": "three"}],
            "c": [{"decision": "OMIT", "question": "no", "answer": "no"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            train, validation = load_examples(path)
            self.assertEqual((train, validation), load_examples(path))
        self.assertEqual(len(train) + len(validation), 3)
        self.assertTrue(all(set(row) == {"question", "answer"} for row in train + validation))
        self.assertEqual(
            len({row["question"][0] for row in train} & {row["question"][0] for row in validation}), 0
        )

    def test_rejects_empty_keep_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps({"a": [{"decision": "KEEP", "question": "q", "answer": ""}]}))
            with self.assertRaisesRegex(ValueError, "empty question or answer"):
                load_examples(path)

    def test_answer_labels_and_truncation_preserve_eos(self):
        tokenizer = Tokenizer()
        rows, truncated = tokenize_examples(
            [{"question": "q", "answer": "x" * MAX_LENGTH}], tokenizer
        )
        prefix = tokenizer.apply_chat_template(
            [{"role": "system", "content": TRAIN_LORA.SYSTEM_PROMPT}, {"role": "user", "content": "q"}],
            tokenize=True, add_generation_prompt=True, return_dict=True,
        )["input_ids"]
        self.assertEqual(truncated, 1)
        self.assertEqual(len(rows[0]["input_ids"]), MAX_LENGTH)
        self.assertEqual(rows[0]["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(rows[0]["labels"][-2:], [99, 42])
        self.assertEqual(rows[0]["input_ids"][-2:], [99, 42])

    def test_system_prompt_precedes_question_in_both_templates(self):
        tokenizer = Tokenizer()
        with patch.object(tokenizer, "apply_chat_template", wraps=tokenizer.apply_chat_template) as apply:
            tokenize_examples([{"question": "q", "answer": "a"}], tokenizer)
        prefix_messages = apply.call_args_list[0].args[0]
        full_messages = apply.call_args_list[1].args[0]
        self.assertEqual(prefix_messages, [
            {"role": "system", "content": TRAIN_LORA.SYSTEM_PROMPT},
            {"role": "user", "content": "q"},
        ])
        self.assertEqual(full_messages, prefix_messages + [{"role": "assistant", "content": "a"}])

    def test_rejects_question_without_room_for_answer(self):
        with self.assertRaisesRegex(ValueError, "question is too long"):
            tokenize_examples([{"question": "q" * MAX_LENGTH, "answer": "a"}], Tokenizer())

    def test_rejects_inconsistent_answer_ending(self):
        tokenizer = Tokenizer()
        original = tokenizer.apply_chat_template

        def malformed(messages, **kwargs):
            result = original(messages, **kwargs)
            if len(messages) == 3 and messages[-1]["content"]:
                result["input_ids"][-1] = 77
            return result

        with patch.object(tokenizer, "apply_chat_template", side_effect=malformed):
            with self.assertRaisesRegex(ValueError, "expected answer ending"):
                tokenize_examples([{"question": "q", "answer": "a"}], tokenizer)

    def test_downloads_only_after_training_to_existing_parent(self):
        examples = [{"question": "q", "answer": "a"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "adapters"
            with (
                patch.object(TRAIN_LORA, "OUTPUT_ROOT", output),
                patch.object(TRAIN_LORA, "load_examples", return_value=(examples, examples)),
                patch.object(TRAIN_LORA.train_adapter, "remote", return_value="run") as remote,
                patch.object(TRAIN_LORA.uuid, "uuid4") as make_id,
                patch.object(TRAIN_LORA.subprocess, "run") as download,
            ):
                make_id.return_value.hex = "abcde12345"
                TRAIN_LORA.main.info.raw_f(run_name="my-adapter")
                remote.assert_called_once_with(examples, examples, "my-adapter-abcde")
                command = download.call_args.args[0]
                self.assertEqual(command[-3:], [TRAIN_LORA.VOLUME_NAME, "my-adapter-abcde", str(output)])
                self.assertTrue(output.is_dir())
                self.assertTrue(download.call_args.kwargs["check"])

    def test_invalid_run_name_fails_before_training(self):
        with patch.object(TRAIN_LORA.train_adapter, "remote") as remote:
            for name in ("", "../adapter", "my adapter", "-adapter"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Run name"):
                    TRAIN_LORA.main.info.raw_f(run_name=name)
            remote.assert_not_called()

    def test_training_failure_does_not_download(self):
        with (
            patch.object(TRAIN_LORA, "load_examples", return_value=([{}], [{}])),
            patch.object(TRAIN_LORA.train_adapter, "remote", side_effect=RuntimeError("failed")),
            patch.object(TRAIN_LORA.subprocess, "run") as download,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                TRAIN_LORA.main.info.raw_f()
            download.assert_not_called()

    def test_download_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(TRAIN_LORA, "OUTPUT_ROOT", Path(directory)),
                patch.object(TRAIN_LORA, "load_examples", return_value=([{}], [{}])),
                patch.object(TRAIN_LORA.train_adapter, "remote"),
                patch.object(TRAIN_LORA.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "modal")),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    TRAIN_LORA.main.info.raw_f()


if __name__ == "__main__":
    unittest.main()
