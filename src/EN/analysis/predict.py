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
model_ckpt = "distilbert-base-uncased"


import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def create_and_train_model(
    train_dataset,
    eval_dataset,
    model,
    batch_size=8,
    num_train_epochs=1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    args = TrainingArguments(
        output_dir="jigsaw",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
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


# Create the trainer and train the model

# You can now use the 'trainer' object for further operations or analysis outside of the function.


def _accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def _compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"accuracy_thresh": _accuracy_thresh(predictions, labels)}
