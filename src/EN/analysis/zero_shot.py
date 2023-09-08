"""Functions for applying the zero-shot classification."""

import random

import torch
from datasets import Dataset
from transformers import AutoTokenizer, pipeline

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def zero_shot_classifier(data):
    """Classify the zero-shot data to receive the classes.

    Args:
        data (datasets.Dataset): The dataset to be classified.

    Returns:
        zero_shot_data (datasets.Dataset): The classified dataset.

    """
    data = _zero_shot_labelling(data)
    model_name = "valhalla/distilbart-mnli-12-6"
    classes = ["labor supply", "labor demand", "government intervention"]
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        multi_label=True,
        device="cuda:0" if torch.cuda.is_available() else None,
    )
    zero_shot_data = classifier(
        data["Description"],
        classes,
        tokenizer=_tokenize,
    )
    zero_shot_data = _transform_to_disk(zero_shot_data)
    return zero_shot_data


def _transform_to_disk(data):
    """Transform datatype to dataset and rename the data.

    Arguments:
        data (dataset.dict): The data to be transformed.

    Returns:
        data (dataset): The transformed data.

    """
    data = {
        "sequence": [item["sequence"] for item in data],
        "classes": [item["labels"] for item in data],
        "label": [item["scores"] for item in data],
    }
    data = Dataset.from_dict(data)
    return data


def _zero_shot_labelling(data):
    """Load the model for zero-shot classification and apply on the data.

    Arguments:
        data (dataset): The data to be tokenized.

    Returns:
        data (dataset): The tokenized data.

    """
    model_name = "valhalla/distilbart-mnli-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df_encoded = data.map(
        lambda batch: _tokenize(batch, tokenizer),
        batched=True,
        batch_size=8,
    )
    return df_encoded


def _tokenize(batch, tokenizer):
    """Define the tokenizer arguments.

    Arguments:
        batch (dataset): The data to be tokenized.
        tokenizer (tokenizer): The tokenizer to be used.

    Returns:
        tokenizer (tokenizer): The tokenized data.

    """
    return tokenizer(
        batch["Description"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
