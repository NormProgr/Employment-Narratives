"Functions for evaluating the zero-shot classification."


def _zero_one_translation(dataset):
    """Translate the labels to 0 and 1.

    Args:
        dataset (datasets.DatasetDict): The dataset to be translated from continuous to binary.

    Returns:
        dataset (datasets.DatasetDict): The translated dataset.

    """
    return dataset.map(
        lambda example: {"label": [1 if x > 0.5 else 0 for x in example["label"]]},
    )
