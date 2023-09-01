"""Tasks running the core analyses."""

import random

import pandas as pd
import pytask
import torch

from EN.analysis.model import fit_logit_model, load_model
from EN.analysis.predict import predict_prob_by_age
from EN.analysis.zero_shot import zero_shot_classifier
from EN.config import BLD, GROUPS, SRC
from EN.utilities import read_yaml

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
@pytask.mark.produces(BLD / "python" / "labelled" / "data_labelled")
def task_fit_model_python(depends_on, produces):
    "Fit a logistic regression model (Python version)."
    data = load_from_disk(depends_on)
    data = data[:100]  # for testing
    zero_shot_classifier(data)
    pass


@pytask.mark.skip
@pytask.mark.depends_on(
    {
        "scripts": ["model.py", "predict.py"],
        "data": BLD / "python" / "data" / "data_clean.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "models" / "model.pickle")
def task_fit_model_python(depends_on, produces):
    "Fit a logistic regression model (Python version)."
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["data"])
    model = fit_logit_model(data, data_info, model_type="linear")
    model.save(produces)


# @pytask.mark.skip
for group in GROUPS:
    kwargs = {
        "group": group,
        "produces": BLD / "python" / "predictions" / f"{group}.csv",
    }

    @pytask.mark.depends_on(
        {
            "data": BLD / "python" / "data" / "data_clean.csv",
            "model": BLD / "python" / "models" / "model.pickle",
        },
    )
    @pytask.mark.skip
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_predict_python(depends_on, group, produces):
        "Predict based on the model estimates (Python version)."
        model = load_model(depends_on["model"])
        data = pd.read_csv(depends_on["data"])
        predicted_prob = predict_prob_by_age(data, model, group)
        predicted_prob.to_csv(produces, index=False)
