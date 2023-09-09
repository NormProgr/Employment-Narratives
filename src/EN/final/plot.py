"""Functions plotting results."""

import pandas as pd
from pandas import json_normalize


def table_produce(data1, data2):
    """Produce the table for the results.

    Args:
        data1 (dict): The data to be transformed.
        data2 (dict): The data to be transformed.

    Returns:
        data (dataframe): The transformed data.
        data2 (dataframe): The transformed data.

    """
    data = json_normalize(data1)
    data2 = json_normalize(data2)
    data2 = pd.DataFrame(data2)
    data = pd.DataFrame(data)
    return data, data2
