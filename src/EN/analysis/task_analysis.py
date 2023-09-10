"""Tasks running the core analyses."""

import pickle
import random

import pandas as pd
import pytask
import torch
from datasets import load_from_disk
from torch import cuda

from EN.analysis.model import bert_model
from EN.analysis.predict import create_and_train_model
from EN.analysis.train_test import create_train_test
from EN.analysis.zero_shot import zero_shot_classifier
from EN.analysis.zero_shot_eval import calculate_accuracy_scores
from EN.config import BLD, SRC
from EN.utilities import read_yaml

# control randomness and set device
model_config = read_yaml(SRC / "analysis" / "model_config.yaml")
device = "cuda" if cuda.is_available() else "cpu"

seed = model_config["seed"]
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


@pytask.mark.depends_on(
    {
        "scripts": ["zero_shot.py"],
        "data": BLD / "python" / "data" / "data_clean",
        "model_config": SRC / "analysis" / "model_config.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "labelled" / "data_labelled")
def task_zero_shot(depends_on, produces):
    "Zero-shot classification that produces the labels for the data."
    model_config = read_yaml(depends_on["model_config"])

    data = load_from_disk(
        depends_on["data"],
    )  # keep an eye of cache data being produced
    if device == "cuda":
        # Handle GPU-specific operations
        labeled_data = zero_shot_classifier(data, model_config)
    else:
        # Handle CPU operations, selecting only 100 data points
        total_examples = len(
            data,
        )  # Replace "train" with the split you want to use (e.g., "test", "validation").
        random_indices = random.sample(range(total_examples), 1000)
        first_100_entries = data.select(random_indices)
        labeled_data = zero_shot_classifier(first_100_entries, model_config)
    labeled_data.save_to_disk(produces)


@pytask.mark.depends_on(
    {
        "scripts": ["zero_shot.py"],
        "data": BLD / "python" / "labelled" / "data_labelled",
        "hand_class": SRC / "data" / "seed_42_classification.csv",
    },
)
@pytask.mark.produces(BLD / "python" / "results" / "acc_scores_zer_shot.pkl")
def task_zero_shot_eval(depends_on, produces):
    "Zero-shot classification that produces the labels for the data."
    data = load_from_disk(depends_on["data"])
    data = pd.DataFrame(data)
    hand_class = pd.read_csv(depends_on["hand_class"])
    acc_scores_zer_shot = calculate_accuracy_scores(hand_class, data)
    with open(produces, "wb") as f:
        pickle.dump(acc_scores_zer_shot, f)


@pytask.mark.depends_on(
    {
        "scripts": ["train_test.py"],
        "data": BLD / "python" / "labelled" / "data_labelled",
    },
)
@pytask.mark.produces(BLD / "python" / "TrainTest" / "TrainTest_data")
def task_TrainTest(depends_on, produces):
    "Separate the data into test and training data."
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
        "model_config": SRC / "analysis" / "model_config.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "model" / "data_model.pkl")
def task_model(depends_on, produces):
    model_config = read_yaml(depends_on["model_config"])
    ds = load_from_disk(depends_on["data"])
    model = bert_model(ds, model_config)
    with open(produces, "wb") as f:
        pickle.dump(model, f)


@pytask.mark.depends_on(
    {
        "scripts": ["model.py"],  # war vorher Train_test und hat geklappt
        "data": BLD / "python" / "model" / "data_model.pkl",
        "model_config": SRC / "analysis" / "model_config.yaml",
    },
)
@pytask.mark.produces(BLD / "python" / "results" / "predict_to_save.pkl")
def task_predict(depends_on, produces):
    """Predict the labels for the data."""
    model_config = read_yaml(depends_on["model_config"])

    with open(depends_on["data"], "rb") as f:
        loaded_data_model = pickle.load(f)
    dataset = loaded_data_model[0]
    model = loaded_data_model[1]
    trainer = create_and_train_model(
        dataset["train_dataset"],
        dataset["val_dataset"],
        model,
        model_config,
    )
    trained = trainer.train()
    evaluated = trainer.evaluate()
    models_to_save = {
        "trained": trained,
        "eval": evaluated,
    }
    with open(produces, "wb") as f:
        pickle.dump(models_to_save, f)
