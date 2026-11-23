import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from historical_time_period import TextDatingModel


class TextDatingModelTests(unittest.TestCase):
    @patch("historical_time_period.AutoModelForSequenceClassification")
    @patch("historical_time_period.AutoTokenizer")
    def test_predicts_decade_from_highest_logit(self, tokenizer_class, model_class):
        tokenizer = tokenizer_class.from_pretrained.return_value
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        model = model_class.from_pretrained.return_value
        model.eval.return_value = model
        logits = torch.zeros(1, 33)
        logits[0, 32] = 5
        model.return_value = SimpleNamespace(logits=logits)

        predictor = TextDatingModel()
        self.assertEqual(predictor.predict_decade("A sentence."), 2020)
        tokenizer.assert_called_once_with(
            "A sentence.", truncation=True, return_tensors="pt"
        )

    @patch("historical_time_period.AutoModelForSequenceClassification")
    @patch("historical_time_period.AutoTokenizer")
    def test_rejects_empty_text(self, tokenizer_class, model_class):
        predictor = TextDatingModel()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            predictor.predict_decade("  ")
        tokenizer_class.from_pretrained.return_value.assert_not_called()


if __name__ == "__main__":
    unittest.main()
