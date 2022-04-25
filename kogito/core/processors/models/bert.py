import torch
import numpy as np
from torch.utils.data import Dataset
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

from kogito.core.processors.models.utils import Evaluator

class BERTHeadDataset(Dataset):
    def __init__(self, df, tokenizer_type="uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(f'bert-base-{tokenizer_type}')
        self.labels = np.asarray(df['label'].to_list())
        self.texts = [self.tokenizer(text, padding='max_length', max_length=32, truncation=True,
                                     return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BERTClassifier(Evaluator, pl.LightningModule):
    def __init__(self, num_classes=3, dropout=0.5, learning_rate=1e-4, freeze_emb=False, model_type="uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(f'bert-base-{model_type}')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)

        if freeze_emb:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False
            self.classifier = nn.Sequential(self.linear)
        else:
            self.classifier = nn.Sequential(self.dropout, self.linear)

        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters()
    
    def forward(self, input_ids, mask):
        _, outputs = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
        outputs = self.classifier(outputs)
        return outputs

    def predict(self, input_ids, mask):
        return F.sigmoid(self.forward(input_ids, mask))
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        mask = X['attention_mask']
        input_ids = X['input_ids'].squeeze(1)
        outputs = self.forward(input_ids, mask)
        train_loss = self.criterion(outputs, y.float())
        preds = F.sigmoid(outputs)
        self.log("train_loss", train_loss, on_epoch=True)
        self.log_metrics(preds, y, type="train")
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        mask = X['attention_mask']
        input_ids = X['input_ids'].squeeze(1)
        outputs = self.forward(input_ids, mask)
        val_loss = self.criterion(outputs, y.float())
        preds = F.sigmoid(outputs)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log_metrics(preds, y, type="val")
        return val_loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        mask = X['attention_mask']
        input_ids = X['input_ids'].squeeze(1)
        outputs = self.forward(input_ids, mask)
        test_loss = self.criterion(outputs, y.float())
        preds = F.sigmoid(outputs)
        self.log("test_loss", test_loss, on_epoch=True)
        self.log_metrics(preds, y, type="test")
        return test_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer