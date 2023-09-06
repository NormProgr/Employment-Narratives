import datasets


def create_train_test(df):
    # Split the dataset using the split_dataset function
    split_data = _split_dataset(df)

    # Create a DatasetDict containing train, validation, and test datasets
    combined_dataset = datasets.DatasetDict(split_data)

    return combined_dataset


def _split_dataset(df):
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

    # Rename columns if needed

    # You may need to specify the 'classes' column name if it's different
    # Assuming it's 'classes' in your dataset, rename it to 'classes'

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
