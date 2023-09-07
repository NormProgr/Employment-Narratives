"""Tasks running the core analyses."""

import random

import pytask
import torch
from datasets import load_from_disk
from torch import cuda

from EN.analysis.model import bert_model
from EN.analysis.predict import create_and_train_model
from EN.analysis.train_test import create_train_test
from EN.analysis.zero_shot import zero_shot_classifier
from EN.config import BLD

device = "cuda" if cuda.is_available() else "cpu"

model_ckpt = "distilbert-base-uncased"


import torch

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import pickle


# if GPU then then switch from data_labelled_subset to classified_data
# make sure classified_data is the right format


@pytask.mark.depends_on(
    {
        "scripts": ["zero_shot.py"],
        "data": BLD / "python" / "data" / "data_clean",
    },
)
@pytask.mark.produces(BLD / "python" / "labelled" / "data_labelled_subset")
def task_zero_shot(depends_on, produces):
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
def task_TrainTest(depends_on, produces):
    "Zero-shot classification that produces the labels for the data."
    data = load_from_disk(
        depends_on["data"],
    )
    data = create_train_test(data)
    data.save_to_disk(produces)
    # fix this then model is easy, just need to add attention and input afterwards


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],  # war vorher Train_test und hat geklappt
        "data": BLD / "python" / "TrainTest" / "TrainTest_data",
    },
)
@pytask.mark.produces(BLD / "python" / "model" / "data_model.pkl")
def task_model(depends_on, produces):
    ds = load_from_disk(depends_on["data"])
    model = bert_model(ds)
    with open(produces, "wb") as f:
        pickle.dump(model, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],  # war vorher Train_test und hat geklappt
        "data": BLD / "python" / "model" / "data_model.pkl",
    },
)
@pytask.mark.produces(BLD / "python" / "predict" / "predict_to_save.pkl")
def task_predict(depends_on, produces):
    with open(depends_on["data"], "rb") as f:
        loaded_data_model = pickle.load(f)
    dataset = loaded_data_model[0]
    model = loaded_data_model[1]
    trainer = create_and_train_model(
        dataset["train_dataset"],
        dataset["val_dataset"],
        model,
    )
    trained = trainer.train()
    evaluated = trainer.evaluate()
    models_to_save = {
        "trained": trained,
        "eval": evaluated,
    }
    with open(produces, "wb") as f:
        pickle.dump(models_to_save, f)
    # need to fix jigsaw and output directory
