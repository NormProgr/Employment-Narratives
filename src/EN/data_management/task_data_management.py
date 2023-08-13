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
    api.dataset_download_files(dataset)  # cnn_zip =
    with zipfile.ZipFile("cnn-articles-after-basic-cleaning.zip", "r") as zip_ref:
        zip_ref.extractall(produces)


# with zipfile.ZipFile('my_archive.zip', 'w') as zipf:
# Add files to the archive


# with zipfile.ZipFile(cnn_zip, 'r') as zip_ref:


# with open(cnn_zip, "w") as output_file:


# with zipfile.ZipFile("cnn-articles-after-basic-cleaning.zip", "r") as cnn_zip:


@pytask.mark.skip
@pytask.mark.depends_on(
    {
        "scripts": ["clean_data.py"],
        "data_info": SRC / "data_management" / "data_info.yaml",
        "data": SRC / "data" / "data.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "data" / "data_clean.csv")
def task_clean_data_python(depends_on, produces):
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    data = clean_data(data, data_info)
    data.to_csv(produces, index=False)
