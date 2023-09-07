import datasets


def create_train_test(df):
    """Create a train, validation, and test dataset."""
    # Split the dataset using the split_dataset function
    split_data = _split_dataset(df)

    # Create a DatasetDict containing train, validation, and test datasets
    combined_dataset = datasets.DatasetDict(split_data)
    combined_dataset = _zero_one_translation(combined_dataset)
    return combined_dataset


def _split_dataset(df):
    """Split the dataset into train, validation, and test datasets."""
    # Shuffle the dataset to ensure randomization
    df = df.shuffle(seed=42)

    # Calculate the split sizes
    total_size = len(df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    total_size - train_size - val_size

    # Split the dataset
    train_dataset = datasets.Dataset.from_dict(df[:train_size])
    val_dataset = datasets.Dataset.from_dict(df[train_size : train_size + val_size])
    test_dataset = datasets.Dataset.from_dict(df[train_size + val_size :])

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


def _zero_one_translation(dataset):
    """Translate the labels to 0 and 1."""
    return dataset.map(
        lambda example: {"label": [1 if x > 0.5 else 0 for x in example["label"]]},
    )
