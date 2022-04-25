import numpy as np
import torch
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import spacy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from kogito.core.processors.models.utils import Evaluator, text_to_embedding


class SWEMHeadDataset(Dataset):
    def __init__(self, df, vocab, embedding_matrix=None, apply_pooling=False, pooling="avg", lang=None):
        if not lang:
            lang = spacy.load("en_core_web_sm")
        self.texts = []

        if apply_pooling:
            # Apply pooling directly without padding
            self.labels = []
            self.features = []

            for index, text in enumerate(df['text']):
                embedding = text_to_embedding(text, vocab=vocab, embedding_matrix=embedding_matrix, lang=lang)
                if embedding is not None:
                    self.features.append(embedding)
                    self.labels.append(df['label'][index])
                    self.texts.append(text)
            
            self.labels = np.asarray(self.labels)
        else:
            # Pad sequences
            self.texts = df['text']
            self.labels = np.asarray(df['label'].to_list()) if 'label' in df.columns else None
            self.features = pad_sequence([torch.tensor([vocab.get(token.text, 1) for token in lang(text)], dtype=torch.int) for text in df['text']],
                                    batch_first=True)


class MaxPool(nn.Module):
    def forward(self, X):
        values, _ = torch.max(X, dim=1)
        return values


class AvgPool(nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=1)


class SWEMClassifier(Evaluator, pl.LightningModule):
    def __init__(self, num_classes=3, pooling="avg", freeze_emb=False, learning_rate=1e-4):
        super(SWEMClassifier, self).__init__()
        embedding_matrix = np.load("data/embedding_matrix_glove_100d.npy", allow_pickle=True)
        self.embedding = nn.Embedding(num_embeddings=embedding_matrix.shape[0],
                                      embedding_dim=embedding_matrix.shape[1]).from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze_emb)
        self.pool = MaxPool() if pooling == "max" else AvgPool()
        self.linear = nn.Linear(embedding_matrix.shape[1], num_classes)
        self.model = nn.Sequential(self.embedding, self.pool, self.linear)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
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