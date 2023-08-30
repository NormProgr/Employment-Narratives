"""Functions for fitting the regression model."""

import torch
from transformers import AutoTokenizer, pipeline


def zero_shot_classifier(data):
    """Classify the zero-shot data to receive the labels."""
    data = _zero_shot_labelling(data)
    model_name = "facebook/bart-large-mnli"
    candidate_labels = ["labor supply", "labor demand", "government intervention"]
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        multi_label=True,
        device="cuda:0" if torch.cuda.is_available() else None,
    )
    zero_shot_data = classifier(
        data["Article text"],
        candidate_labels,
        tokenizer=_tokenize,
    )
    return zero_shot_data


def _zero_shot_labelling(data):
    """Load the model for zero-shot classification and apply on the data."""
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df_encoded = data.map(
        lambda batch: _tokenize(batch, tokenizer),
        batched=True,
        batch_size=None,
    )
    return df_encoded


def _tokenize(batch, tokenizer):
    """Define the tokenizer."""
    return tokenizer(batch["Article text"], padding="max_length", truncation=True)
