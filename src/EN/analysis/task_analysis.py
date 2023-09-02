"""Tasks running the core analyses."""

import json
import random

import pytask
import torch
from datasets import load_from_disk

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
@pytask.mark.produces(BLD / "python" / "labelled" / "data_labelled_subset.json")
def task_fit_model_python(depends_on, produces):
    "Fit a logistic regression model (Python version)."
    data = load_from_disk(depends_on["data"])
    first_100_entries = data.select(range(100))
    first_100_entries = zero_shot_classifier(first_100_entries)
    with open(produces, "w") as json_file:
        json.dump(first_100_entries, json_file, indent=4)
