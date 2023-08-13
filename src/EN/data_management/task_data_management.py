"""Tasks for managing the data."""
import zipfile

import pandas as pd
import pytask

from EN.config import BLD, SRC
from EN.data_management import authenticate_to_kaggle, clean_data
from EN.utilities import read_yaml


@pytask.mark.produces(BLD / "python" / "data" / "cnn-articles-after-basic-cleaning.zip")
def task_load_data_python(produces):
    """Clean the data (Python version)."""
    api = authenticate_to_kaggle()
    dataset = "hadasu92/cnn-articles-after-basic-cleaning"
    api.dataset_download_files(dataset)
    with zipfile.ZipFile("cnn-articles-after-basic-cleaning.zip", "r") as zip_ref:
        zip_ref.extractall(produces)


@pytask.mark.skip
@pytask.mark.depends_on(SRC / "data" / "cepr_march.zip")
@pytask.mark.produces(BLD / "python" / "data")
def task_unzip(depends_on, produces):
    """Task for unzipping data."""
    with zipfile.ZipFile(depends_on, "r") as zip_ref:
        zip_ref.extractall(produces)


@pytask.mark.skip
@pytask.mark.produces(BLD / "python" / "data" / "cnn-articles-after-basic-cleaning.zip")
def task_load_data_python(produces):
    """Clean the data (Python version)."""
    # kaggle datasets download -d hadasu92/cnn-articles-after-basic-cleaning
    cnn_zip = os.system(
        'kaggle datasets download -d "hadasu92/cnn-articles-after-basic-cleaning"',
    )
    with open(produces, "wb") as output_file:
        output_file.write(cnn_zip)

    #    if cnn_zip is None:
    #        with open(produces, "wb") as output_file:


@pytask.mark.skip
@pytask.mark.depends_on(SRC / "data" / "cepr_march.zip")
@pytask.mark.produces(BLD / "python" / "data")
def task_unzip(depends_on, produces):
    """Task for unzipping data."""
    with zipfile.ZipFile(depends_on, "r") as zip_ref:
        zip_ref.extractall(produces)


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
    """Clean the data (Python version)."""
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    data = clean_data(data, data_info)
    data.to_csv(produces, index=False)
