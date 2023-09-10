import datasets


def create_train_test(df):
    """Create a train, validation, and test dataset.

    Args:
        df (pandas.DataFrame): The dataframe to be split.

    Returns:
        combined_dataset (datasets.DatasetDict): The split dataset.

    """  # need to delete the validation data
    split_data = _split_dataset(df)

    combined_dataset = datasets.DatasetDict(split_data)
    combined_dataset = _zero_one_translation(combined_dataset)
    return combined_dataset


def _split_dataset(df):
    """Split and shuffle the dataset into train, validation, and test datasets.

    Args:
        df (pandas.DataFrame): The dataframe to be split in a train, validation, and test dataset.

    Returns:
        train_dataset (datasets.Dataset): The training dataset.
        val_dataset (datasets.Dataset): The validation dataset.
        test_dataset (datasets.Dataset): The test dataset.

    """  # delete the validation data
    df = df.shuffle(seed=42)  # change this

    total_size = len(df)
    train_size = int(0.8 * total_size)
    val_size = int(0.2 * total_size)
    total_size - train_size - val_size

    train_dataset = datasets.Dataset.from_dict(df[:train_size])
    val_dataset = datasets.Dataset.from_dict(df[train_size : train_size + val_size])

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


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
