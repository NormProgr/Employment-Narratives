"""Tasks for managing the data."""

import zipfile

import pandas as pd
import pytask

from EN.config import BLD, SRC
from EN.data_management import authenticate_to_kaggle, clean_data, select_random_entries
from EN.utilities import read_yaml


# for dataset in sets:


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
        / "CNN_Articels_clean.csv",  # / f"{dataset}"  / "CNN_Articels_clean.csv"
        "file2": BLD
        / "python"
        / "data"
        / "CNN_Articels_clean_2"
        / "CNN_Articels_clean.csv",
    },
)  # / "cnn-articles-after-basic-cleaning.zip"
def task_load_data(produces):
    """Clean the data (Python version)."""
    api = authenticate_to_kaggle()
    dataset = "hadasu92/cnn-articles-after-basic-cleaning"
    api.dataset_download_files(dataset)
    with zipfile.ZipFile("cnn-articles-after-basic-cleaning.zip", "r") as zip_ref:
        zip_ref.extractall(produces["file"])
    # kaggle.api.dataset_download_files(
    #    dataset,


@pytask.mark.depends_on(
    {
        "scripts": ["clean_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "Article_1": BLD / "python" / "data"
        # / "cnn-articles-after-basic-cleaning.zip"
        / "CNN_Articels_clean" / "CNN_Articels_clean.csv",
        "Article_2": BLD / "python" / "data"
        # / "cnn-articles-after-basic-cleaning.zip"
        / "CNN_Articels_clean_2" / "CNN_Articels_clean.csv",
        "Seed42_hand_classification": SRC / "data" / "seed_42_classification.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_clean")  # {dataset}
def task_clean_data_python(depends_on, produces):
    "Clean the data from unwanted categories and concetenate the raw files. Also produces evaluation set."
    df_1 = pd.read_csv(depends_on["Article_1"])  # need to delete cache here
    df_2 = pd.read_csv(
        depends_on["Article_2"],
    )
    data_info = read_yaml(depends_on["data_info"])
    data = clean_data(df_1, df_2, data_info)
    data.save_to_disk(produces)


@pytask.mark.depends_on(
    {
        "scripts": ["clean_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "Article_1": BLD / "python" / "data"
        # / "cnn-articles-after-basic-cleaning.zip"
        / "CNN_Articels_clean" / "CNN_Articels_clean.csv",
        "Article_2": BLD / "python" / "data"
        # / "cnn-articles-after-basic-cleaning.zip"
        / "CNN_Articels_clean_2" / "CNN_Articels_clean.csv",
        "Seed42_hand_classification": SRC / "data" / "seed_42_classification.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "benchmark.csv")  # {dataset}
def task_select_data(depends_on, produces):
    "Subset the data to 50 entries and add the hand classification."
    df_1 = pd.read_csv(depends_on["Article_1"])  # need to delete cache here
    df_2 = pd.read_csv(
        depends_on["Article_2"],
    )  # sometimes it works, sometimes it doesn't
    data_info = read_yaml(depends_on["data_info"])
    data = clean_data(df_1, df_2, data_info)
    data = select_random_entries(data, num_entries=50, random_state=42)
    random_seed = pd.read_csv(
        depends_on["Seed42_hand_classification"],
    )  # concetanete function and then done
    hand_class = pd.read_csv(depends_on["Seed42_hand_classification"])
    data = pd.concat([random_seed, hand_class], ignore_index=True)
    data.to_csv(produces)
