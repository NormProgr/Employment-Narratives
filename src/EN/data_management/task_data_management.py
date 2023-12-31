"""Tasks for managing the data."""

import zipfile

import pandas as pd
import pytask

from EN.config import BLD, SRC
from EN.data_management import authenticate_to_kaggle, clean_data
from EN.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "scripts": ["load_data.py"],
    },
)
@pytask.mark.task
@pytask.mark.produces(
    {
        "file": BLD / "python" / "data",
        "file1": BLD
        / "python"
        / "data"
        / "CNN_Articels_clean"
        / "CNN_Articels_clean.csv",
        "file2": BLD
        / "python"
        / "data"
        / "CNN_Articels_clean_2"
        / "CNN_Articels_clean.csv",
    },
)
def task_load_data(produces):
    """Load the data from kaggle."""
    api = authenticate_to_kaggle()
    dataset = "hadasu92/cnn-articles-after-basic-cleaning"
    api.dataset_download_files(dataset)
    with zipfile.ZipFile("cnn-articles-after-basic-cleaning.zip", "r") as zip_ref:
        zip_ref.extractall(produces["file"])


@pytask.mark.depends_on(
    {
        "scripts": ["clean_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "Article_1": BLD
        / "python"
        / "data"
        / "CNN_Articels_clean"
        / "CNN_Articels_clean.csv",
        "Article_2": BLD
        / "python"
        / "data"
        / "CNN_Articels_clean_2"
        / "CNN_Articels_clean.csv",
        "Seed42_hand_classification": SRC / "data" / "seed_42_classification.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_clean")
def task_clean_data_python(depends_on, produces):
    "Clean the data concetenate the raw files. Also produces evaluation dataset."
    df_1 = pd.read_csv(depends_on["Article_1"])  # need to delete cache here
    df_2 = pd.read_csv(
        depends_on["Article_2"],
    )
    data_info = read_yaml(depends_on["data_info"])
    data = clean_data(df_1, df_2, data_info)
    data.save_to_disk(produces)
