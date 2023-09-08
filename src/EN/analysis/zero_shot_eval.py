"Functions that evaluate the zero-shot performance of the model."

from datasets import Dataset, DatasetDict


def benchmark_zero_shot_classifier(data):
    data = _pd_to_dataset(data)
    return data


def _pd_to_dataset(data):
    """Convert pandas DataFrame to HuggingFace Dataset.

    Returns:
        torch_data (dataset): HuggingFace Dataset ready for text classification.

    """
    data = Dataset.from_pandas(data)
    dataset_dict = DatasetDict({"my_dataset": data})
    torch_data = dataset_dict["my_dataset"]
    return torch_data
