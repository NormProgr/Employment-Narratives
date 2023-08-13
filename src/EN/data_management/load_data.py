"""Function(s) for cleaning the data set(s)."""
from kaggle.api.kaggle_api_extended import KaggleApi


def authenticate_to_kaggle():
    """Authenticate to Kaggle."""
    api = KaggleApi()
    api.authenticate()
    return api
