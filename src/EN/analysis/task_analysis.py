"""Tasks running the core analyses."""

import random

import pytask
import torch
from datasets import load_from_disk

from EN.analysis.train_test import create_train_test
from EN.analysis.zero_shot import zero_shot_classifier
from EN.config import BLD

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# if GPU then then switch from data_labelled_subset to classified_data
# make sure classified_data is the right format


@pytask.mark.depends_on(
    {
        "scripts": ["zero_shot.py"],
        "data": BLD / "python" / "data" / "data_clean",
    },
)
@pytask.mark.produces(BLD / "python" / "labelled" / "data_labelled_subset")
def task_fit_model_python(depends_on, produces):
    "Zero-shot classification that produces the labels for the data."
    data = load_from_disk(
        depends_on["data"],
    )  # keep an eye of cache data being produced
    first_100_entries = data.select(range(100))
    first_100_entries = zero_shot_classifier(first_100_entries)
    first_100_entries.save_to_disk(produces)
    # fix this then model is easy, just need to add attention and input afterwards


@pytask.mark.depends_on(
    {
        "scripts": ["train_test.py"],
        "data": BLD / "python" / "labelled" / "data_labelled_subset",
    },
)
@pytask.mark.produces(BLD / "python" / "TrainTest" / "TrainTest_data")
def task_fit_model_python(depends_on, produces):
    "Zero-shot classification that produces the labels for the data."
    data = load_from_disk(
        depends_on["data"],
    )
    data = create_train_test(data)
    data.save_to_disk(produces)
    # fix this then model is easy, just need to add attention and input afterwards
