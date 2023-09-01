"""Tasks for managing the data."""
import zipfile

import pandas as pd
import pytask

from EN.config import BLD, SRC
from EN.data_management import authenticate_to_kaggle, clean_data
from EN.utilities import read_yaml


@pytask.mark.produces(BLD / "python" / "data" / "cnn-articles-after-basic-cleaning.zip")
def task_load_data_python(produces):
    """Clean the data (Python version).

    Download needs up to 5 minutes. Is this due to internet or coding issue?

    """
    api = authenticate_to_kaggle()
    dataset = "hadasu92/cnn-articles-after-basic-cleaning"
    api.dataset_download_files(dataset)
    with zipfile.ZipFile("cnn-articles-after-basic-cleaning.zip", "r") as zip_ref:
        zip_ref.extractall(produces)


@pytask.mark.depends_on(
    {
        "scripts": ["clean_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "Article_1": BLD
        / "python"
        / "data"
        / "cnn-articles-after-basic-cleaning.zip"
        / "CNN_Articels_clean"
        / "CNN_Articels_clean.csv",
        "Article_2": BLD
        / "python"
        / "data"
        / "cnn-articles-after-basic-cleaning.zip"
        / "CNN_Articels_clean_2"
        / "CNN_Articels_clean.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_clean")
# @pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
def task_clean_data_python(depends_on, produces):
    "Clean the data from unwanted categories and concetenate the raw files. Instead of using to_csv we should use to pickle."
    df_1 = pd.read_csv(depends_on["Article_1"])  # need to delete cache here
    df_2 = pd.read_csv(
        depends_on["Article_2"],
    )  # sometimes it works, sometimes it doesn't
    data_info = read_yaml(depends_on["data_info"])
    data = clean_data(df_1, df_2, data_info)
    data.save_to_disk(produces)
