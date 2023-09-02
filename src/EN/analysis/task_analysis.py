"""Tasks running the core analyses."""

import random

import pytask
import torch
from datasets import Dataset, load_from_disk

from EN.analysis.zero_shot import zero_shot_classifier
from EN.config import BLD

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


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
    dataset_dict = {
        "sequence": [item["sequence"] for item in first_100_entries],
        "labels": [item["labels"] for item in first_100_entries],
        "scores": [item["scores"] for item in first_100_entries],
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk(produces)
    # fix this then model is easy, just need to add attention and input afterwards
