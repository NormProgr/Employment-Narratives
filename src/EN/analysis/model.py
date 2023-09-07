"""Functions for fitting the regression model."""

seed = 42
import random

import torch

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from torch import cuda
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

device = "cuda" if cuda.is_available() else "cpu"

model_ckpt = "distilbert-base-uncased"


import torch
from transformers import AutoTokenizer


def bert_model(ds):
    """Produce the encoded data and the model.

    Arguments:
        ds (dataset): The dataset to be encoded.

    Returns:
        ds_encoded (dataset): Encoded huggingface dataset.
        model (model): The model to be trained.

    """
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Tokenize the dataset
    ds_encoded = _tokenize_dataset(ds, tokenizer)
    num_labels = 3
    model = BertForMultilabelSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=num_labels,
    ).to(device)

    return ds_encoded, model


# Function to tokenize the dataset
def _tokenize_dataset(ds, tokenizer):
    """Tokenize the dataset.

    Arguments:
        ds (dataset): The dataset to be encoded.
        tokenizer (tokenizer): The tokenizer to be used.

    Returns:
        ds_encoded (dataset): Encoded huggingface dataset.

    """

    def tokenize(batch):
        return tokenizer(batch["sequence"], padding=True, truncation=True)

    ds_encoded = ds.map(tokenize, batched=True, batch_size=None)
    ds_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    ds_encoded.set_format("torch")
    return ds_encoded


class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    """BertForMultilabelSequenceClassification is a custom extension of the BERT model
    for multilabel sequence classification tasks.

    Args:
        config (BertConfig): The model configuration class specifying the model's
            architecture and hyperparameters.

    Attributes:
        config (BertConfig): The model configuration class.

    """

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Forward pass of the BertForMultilabelSequenceClassification model.

        Args:
            input_ids (Tensor, optional): Input token IDs.
            attention_mask (Tensor, optional): Mask indicating which tokens should
                be attended to.
            token_type_ids (Tensor, optional): Token type IDs.
            position_ids (Tensor, optional): Positional embeddings.
            head_mask (Tensor, optional): Mask to control which heads are used in
                the attention layers.
            inputs_embeds (Tensor, optional): Custom embeddings for input tokens.

            labels (Tensor, optional): Ground truth labels for the classification
                task.
            output_attentions (bool, optional): Whether to output attention weights.

            output_hidden_states (bool, optional): Whether to output hidden states.

            return_dict (bool, optional): Whether to return outputs as a dictionary.


        Returns:
            SequenceClassifierOutput: Returns a dictionary containing model outputs, including logits and optional additional outputs.

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()  # probably warning source
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.float().view(-1, self.num_labels),
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss, *output)) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
