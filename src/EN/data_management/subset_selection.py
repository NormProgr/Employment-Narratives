"Take a random subset of the data."

import numpy as np
import pandas as pd


def select_random_entries(dataframe, num_entries=50, random_state=42):
    """Select a random set of entries from a Pandas DataFrame.

    Parameters:
        dataframe (json): The input DataFrame with 6 columns.
        num_entries (int): The number of random entries to select (default is 50).
        random_state (int or None): Random seed for reproducibility (default is None).

    Returns:
        random_entries (pd.DataFrame): A DataFrame containing exactly 50 randomly selected entries.

    """
    dataframe = pd.DataFrame(dataframe)

    if random_state is not None:
        np.random.seed(random_state)

    if num_entries > len(dataframe):
        raise ValueError(
            "Number of entries to select cannot exceed the total number of rows.",
        )

    if len(dataframe) <= num_entries:
        random_entries = dataframe
    else:
        random_indices = np.random.choice(
            dataframe["__index_level_0__"],
            size=num_entries,
            replace=False,
        )
        random_entries = dataframe[dataframe["__index_level_0__"].isin(random_indices)]

    while len(random_entries) < num_entries:
        additional_indices = np.random.choice(
            dataframe["__index_level_0__"],
            size=num_entries - len(random_entries),
            replace=False,
        )
        additional_entries = dataframe[
            dataframe["__index_level_0__"].isin(additional_indices)
        ]
        random_entries = pd.concat([random_entries, additional_entries])

    return random_entries.sample(n=num_entries, random_state=random_state)
