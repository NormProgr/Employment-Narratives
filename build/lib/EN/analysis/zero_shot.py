"""Functions for fitting the regression model."""

import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle


def zero_shot_labelling():
    pass


def load_model(path):
    """Load statsmodels model.

    Args:
        path (str or pathlib.Path): Path to model file.

    Returns:
        statsmodels.base.model.Results: The stored model.

    """
    return load_pickle(path)
