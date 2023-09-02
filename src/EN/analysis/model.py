"""Functions for fitting the regression model."""

from datasets import Dataset


def fit_logit_model(data, data_info, model_type):
    """Fit a logit model to data."""
    data = _test_train_split(data)
    pass


def _test_train_split(df):
    """Split the data into test and training data for consumption."""
    df = df.shuffle(seed=42)  # need to specify the randomness level
    total_size = len(df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    train_dataset = Dataset.from_dict(df[:train_size])
    val_dataset = Dataset.from_dict(df[train_size : train_size + val_size])
    test_dataset = Dataset.from_dict(df[train_size:])

    val_dataset = val_dataset.rename_column("sequence", "text")
    train_dataset = train_dataset.rename_column("sequence", "text")
    test_dataset = test_dataset.rename_column("sequence", "text")

    val_dataset = val_dataset.rename_column("labels", "label")
    train_dataset = train_dataset.rename_column("labels", "label")
    test_dataset = test_dataset.rename_column("labels", "label")
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
