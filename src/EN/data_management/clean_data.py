"""Function(s) for cleaning the data set(s)."""

import pandas as pd
from datasets import Dataset, DatasetDict


def clean_data(data_1, data_2, data_info):
    """Clean data set.

    Information on data columns is stored in ``data_management/data_info.yaml``.

    Args:
        data (pandas.DataFrame): The data set.
        data_info (dict): Information on data set stored in data_info.yaml. The
            following keys can be accessed:
            - 'Index': Running number
            - 'Author': Author who wrote Article
            - 'Date published': Publishing date of Article
            - 'Category': Higher level category of Article
            - 'Section': Lower level category of Article
            - 'url': URL to data set
            - 'Headline': Headline of Article
            - 'Description': Short Summary of Article
            - 'Keywords': Keywords of Article
            - 'Second headline': Second Headline of Article
            - 'Article text': Full article text

    Returns:
        merged_dataset (pandas.DataFrame): The cleaned data set.

    """
    if not set(data_1.columns) == set(data_2.columns):
        raise ValueError("Both datasets must have the same columns.")
    merged_dataset = pd.concat([data_1, data_2], axis=0)
    # put this into task
    merged_dataset = _drop_columns(merged_dataset, data_info)
    merged_dataset = _pd_to_dataset(merged_dataset)
    return merged_dataset


def _drop_columns(data, data_info):
    """Drop columns from data set.

    Args:
        data (pandas.DataFrame): The data set.
        data_info (yaml): List of columns to drop.

    Returns:
        filtered_df (pandas.DataFrame): The data set without the dropped columns.

    """
    data = data.drop(columns=data_info["columns_to_drop"])
    data = data.dropna()
    filtered_df = data[
        ~data[data_info["column_name"]].isin(data_info["values_to_remove"])
    ]

    return filtered_df


def _pd_to_dataset(data):
    """Convert pandas DataFrame to HuggingFace Dataset.

    Returns:
        torch_data (dataset): HuggingFace Dataset ready for text classification.

    """
    data = Dataset.from_pandas(data)
    dataset_dict = DatasetDict({"my_dataset": data})
    torch_data = dataset_dict["my_dataset"]
    return torch_data
