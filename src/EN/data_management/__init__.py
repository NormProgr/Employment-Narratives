"""Functions for managing data."""

from EN.data_management.clean_data import clean_data
from EN.data_management.load_data import authenticate_to_kaggle

__all__ = [clean_data, authenticate_to_kaggle]
