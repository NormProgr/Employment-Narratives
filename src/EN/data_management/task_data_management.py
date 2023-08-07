"""Tasks for managing the data."""
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
    cnn_zip = api.dataset_download_files(dataset)
    with open(produces, "wb") as output_file:
        output_file.write(cnn_zip)


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
