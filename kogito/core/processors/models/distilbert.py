import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    PretrainedConfig,
    PreTrainedModel,
)

from kogito.core.processors.models.utils import Evaluator


class DistilBERTHeadDataset(Dataset):
    def __init__(self, data, tokenizer_type="uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            f"distilbert-base-{tokenizer_type}"
        )
        self.labels = (
            np.asarray(data["label"].to_list())
            if isinstance(data, pd.DataFrame)
            else None
        )
        texts = data["text"] if isinstance(data, pd.DataFrame) else data
        self.features = [
            self.tokenizer(
                text,
                padding="max_length",
                max_length=32,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class DistilBERTConfig(PretrainedConfig):
    def __init__(
        self,
        num_classes=3,
        dropout=0.5,
        learning_rate=1e-4,
        freeze_emb=False,
        model_case="uncased",
        **kwargs,
    ):
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.freeze_emb = freeze_emb
        self.model_case = model_case
        super().__init__(**kwargs)


class DistilBERTClassifier(PreTrainedModel, Evaluator, pl.LightningModule):
    config_class = DistilBERTConfig

    def __init__(self, config: DistilBERTConfig):
        super().__init__(config)
        self.distilbert = DistilBertModel.from_pretrained(
            f"distilbert-base-{config.model_case}"
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(768, config.num_classes)

        if config.freeze_emb:
            for parameter in self.distilbert.parameters():
                parameter.requires_grad = False
            self.classifier = nn.Sequential(self.linear)
        else:
            self.classifier = nn.Sequential(self.dropout, self.linear)

        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = config.learning_rate

        self.save_hyperparameters(config.to_dict(), ignore="config")

    def forward(self, input_ids, mask):
        outputs = self.distilbert(
            input_ids=input_ids, attention_mask=mask, return_dict=False
        )
        outputs = self.classifier(outputs[0][:, 0, :])
        return outputs

    def training_step(self, batch, batch_idx):
        X, y = batch
        mask = X["attention_mask"]
        input_ids = X["input_ids"].squeeze(1)
        outputs = self.forward(input_ids, mask)
        train_loss = self.criterion(outputs, y.float())
        preds = torch.sigmoid(outputs)
        self.log("train_loss", train_loss, on_epoch=True)
        self.log_metrics(preds, y, type="train")
        return train_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        mask = X["attention_mask"]
        input_ids = X["input_ids"].squeeze(1)
        outputs = self.forward(input_ids, mask)
        val_loss = self.criterion(outputs, y.float())
        preds = torch.sigmoid(outputs)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log_metrics(preds, y, type="val")
        return val_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        mask = X["attention_mask"]
        input_ids = X["input_ids"].squeeze(1)
        outputs = self.forward(input_ids, mask)
        test_loss = self.criterion(outputs, y.float())
        preds = torch.sigmoid(outputs)
        self.log("test_loss", test_loss, on_epoch=True)
        self.log_metrics(preds, y, type="test")
        return test_loss

    def predict_step(self, batch, batch_idx):
        X = batch
        mask = X["attention_mask"]
        input_ids = X["input_ids"].squeeze(1)
        outputs = self.forward(input_ids, mask)
        preds = torch.sigmoid(outputs)
        return preds

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
