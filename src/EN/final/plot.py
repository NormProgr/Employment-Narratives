"""Functions plotting results."""

import pandas as pd
from pandas import json_normalize


def table_produce(data1, data2):
    data = json_normalize(data1)
    data2 = json_normalize(data2)
    data2 = pd.DataFrame(data2)
    data = pd.DataFrame(data)
    return data, data2
