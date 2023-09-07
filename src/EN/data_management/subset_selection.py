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
        random_entries (pd.DataFrame): A DataFrame containing the randomly selected entries.

    """
    dataframe = pd.DataFrame(dataframe)

    if random_state is not None:
        np.random.seed(random_state)

    if num_entries > len(dataframe):
        raise ValueError(
            "Number of entries to select cannot exceed the total number of rows.",
        )

    # Use Pandas' sample method to select random entries
    random_entries = dataframe.sample(n=num_entries)

    return random_entries
