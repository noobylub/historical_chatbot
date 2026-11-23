"""Predict the decade of an English sentence with ChangeIsKey/text-dating.

Install dependencies with ``python -m pip install torch transformers``, then run:
``python historical_time_period.py "Your sentence goes here"``

In a Jupyter notebook:
``from historical_time_period import TextDatingModel``
``model = TextDatingModel()``
``model.predict_decade("Your sentence goes here")``

The model was trained on sentences from 1700 to 2021. Long inputs are
truncated to the tokenizer's maximum length; predictions are estimates.
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "ChangeIsKey/text-dating"
DEMO_TEXT = (
    "He put the phone back in the cradle and turned toward the kitchen."
)


class TextDatingModel:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID
        ).eval()

    def predict_decade(self, text: str) -> int:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        inputs = self.tokenizer(
            text,
            truncation=True,
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            predicted_label = logits.argmax(dim=1).item()
            return 1700 + predicted_label * 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "text",
        nargs="?",
        default=DEMO_TEXT,
        help="English sentence to date (defaults to a demo sentence)",
    )
    args = parser.parse_args()

    if not args.text.strip():
        parser.error("text must not be empty")
    decade = TextDatingModel().predict_decade(args.text)
    print(f"Predicted decade: {decade}s")


if __name__ == "__main__":
    main()
