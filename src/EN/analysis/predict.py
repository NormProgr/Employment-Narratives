"""Functions for predicting outcomes based on the estimated model."""
seed = 42
import random

import torch

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from torch import cuda
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

device = "cuda" if cuda.is_available() else "cpu"


def create_and_train_model(train_dataset, eval_dataset, model, model_config):
    """Train the model and define its arguments.

    Args:
        train_dataset (Dataset): Training dataset containing input features and labels.
        eval_dataset (Dataset): Evaluation dataset containing input features and labels.
        model (PreTrainedModel): The pre-trained model to fine-tuning.
        model_config (dict): The model configuration.

    Returns:
        Trainer: A trainer object configured for training the model.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name_pred"])
    args = TrainingArguments(
        output_dir="jigsaw",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=model_config["batch_size"],
        per_device_eval_batch_size=model_config["batch_size"],
        num_train_epochs=model_config["epochs"],
        weight_decay=model_config["weight_decay"],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer


def _accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    """Compute the accuracy score using a threshold for binary classification.

    Args:
        y_pred (tensor): Predicted values.
        y_true(tensor): True labels.
        thresh (float): Threshold for class prediction.
        sigmoid (bool): Apply sigmoid activation to y_pred.

    Returns:
        float: The accuracy score.

    """
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()  # what does this mean?
    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def _compute_metrics(eval_pred):
    """Compute the accuracy score based on evaluation predictions.

    Args:
        eval_pred (list): Evaluation predictions.

    Returns:
        dict: A dictionary containing the accuracy score.

    """
    predictions, labels = eval_pred
    return {"accuracy_thresh": _accuracy_thresh(predictions, labels)}
