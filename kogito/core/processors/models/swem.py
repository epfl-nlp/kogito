import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import spacy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedModel

from kogito.core.processors.models.utils import Evaluator, text_to_embedding


class SWEMHeadDataset(Dataset):
    def __init__(
        self,
        data,
        vocab,
        embedding_matrix=None,
        apply_pooling=False,
        pooling="avg",
        lang=None,
    ):
        texts = data["text"] if isinstance(data, pd.DataFrame) else data
        labels = data["label"] if isinstance(data, pd.DataFrame) else None

        if not lang:
            lang = spacy.load("en_core_web_sm")
        self.texts = []

        if apply_pooling:
            # Apply pooling directly without padding
            self.labels = []
            self.features = []

            for index, text in enumerate(texts):
                embedding = text_to_embedding(
                    text, vocab=vocab, embedding_matrix=embedding_matrix, lang=lang
                )
                if embedding is not None:
                    self.features.append(embedding)
                    if labels is not None:
                        self.labels.append(labels[index])
                    self.texts.append(text)

            self.labels = np.asarray(self.labels)
        else:
            # Pad sequences
            self.texts = texts
            self.labels = np.asarray(labels.to_list()) if labels is not None else None
            self.features = pad_sequence(
                [
                    torch.tensor(
                        [vocab.get(token.text, 1) for token in lang(text)],
                        dtype=torch.int,
                    )
                    for text in texts
                ],
                batch_first=True,
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class MaxPool(nn.Module):
    def forward(self, X):
        values, _ = torch.max(X, dim=1)
        return values


class AvgPool(nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=1)


class SWEMConfig(PretrainedConfig):
    def __init__(
        self,
        num_classes=3,
        pooling="avg",
        freeze_emb=False,
        learning_rate=1e-4,
        num_embeddings=400002,
        embedding_dim=100,
        **kwargs
    ):
        self.num_classes = num_classes
        self.pooling = pooling
        self.freeze_emb = freeze_emb
        self.learning_rate = learning_rate
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        super().__init__(**kwargs)


class SWEMClassifier(PreTrainedModel, Evaluator, pl.LightningModule):
    config_class = SWEMConfig

    def __init__(self, config: SWEMConfig):
        super().__init__(config)

        try:
            embedding_matrix = np.load(
                "data/embedding_matrix_glove_100d.npy", allow_pickle=True
            )
            self.embedding = nn.Embedding(
                num_embeddings=embedding_matrix.shape[0],
                embedding_dim=embedding_matrix.shape[1],
            ).from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=config.freeze_emb,
            )
        except FileNotFoundError:
            self.embedding = nn.Embedding(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
            )

        self.pool = MaxPool() if config.pooling == "max" else AvgPool()
        self.linear = nn.Linear(self.embedding.embedding_dim, config.num_classes)
        self.model = nn.Sequential(self.embedding, self.pool, self.linear)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = config.learning_rate
        self.save_hyperparameters(config.to_dict(), ignore="config")

    def forward(self, X):
        outputs = self.model(X)
        probs = F.sigmoid(outputs)
        return probs

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self.model(X)
        train_loss = self.criterion(outputs, y.float())
        preds = self.forward(X)
        self.log("train_loss", train_loss, on_epoch=True)
        self.log_metrics(preds, y, type="train")
        return train_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self.model(X)
        val_loss = self.criterion(outputs, y.float())
        preds = self.forward(X)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log_metrics(preds, y, type="val")
        return val_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        outputs = self.model(X)
        test_loss = self.criterion(outputs, y.float())
        preds = self.forward(X)
        self.log("test_loss", test_loss, on_epoch=True)
        self.log_metrics(preds, y, type="test")
        return test_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
