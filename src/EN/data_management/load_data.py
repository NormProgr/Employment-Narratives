"""Loading the raw data from kaggle."""
from kaggle.api.kaggle_api_extended import KaggleApi


def authenticate_to_kaggle():
    """Authenticate to Kaggle by sending the API Token."""
    api = KaggleApi()
    api.authenticate()
    return api
