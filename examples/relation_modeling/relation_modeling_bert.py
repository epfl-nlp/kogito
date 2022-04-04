#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[1]:


import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class HeadDataset(Dataset):
    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.labels = np.asarray(df['label'].to_list())
        self.texts = [self.tokenizer(text, padding='max_length', max_length=32, truncation=True,
                                     return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BERTClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, dropout=0.5, learning_rate=1e-4, freeze_emb=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
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
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision(num_classes=3, average='weighted')
        self.val_precision = torchmetrics.Precision(num_classes=3, average='weighted')
        self.train_recall = torchmetrics.Recall(num_classes=3, average='weighted')
        self.val_recall = torchmetrics.Recall(num_classes=3, average='weighted')
        self.train_f1 = torchmetrics.F1Score(num_classes=3, average='weighted')
        self.val_f1 = torchmetrics.F1Score(num_classes=3, average='weighted')
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
        self.train_accuracy(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)
        self.log("train_loss", train_loss, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_epoch=True)
        self.log('train_precision', self.train_precision, on_epoch=True)
        self.log('train_recall', self.train_recall, on_epoch=True)
        self.log('train_f1', self.train_f1, on_epoch=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        mask = X['attention_mask']
        input_ids = X['input_ids'].squeeze(1)
        outputs = self.forward(input_ids, mask)
        val_loss = self.criterion(outputs, y.float())
        preds = F.sigmoid(outputs)
        self.val_accuracy(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", val_loss, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, on_epoch=True)
        self.log('val_precision', self.val_precision, on_epoch=True)
        self.log('val_recall', self.val_recall, on_epoch=True)
        self.log('val_f1', self.val_f1, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# In[3]:


from relation_modeling_utils import load_data

train_df = load_data("data/atomic2020_data-feb2021/train.tsv", multi_label=True)
dev_df = load_data("data/atomic2020_data-feb2021/dev.tsv", multi_label=True)
train_data = HeadDataset(train_df)
val_data = HeadDataset(dev_df)


# In[4]:


len(train_data), len(val_data)


# In[4]:


train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=2)


# In[5]:

from relation_modeling_utils import get_timestamp
from pytorch_lightning.loggers import WandbLogger
import wandb

timestamp = get_timestamp()
wandb_logger = WandbLogger(project="kogito-relation-matcher", name=f"bert_multi_label_finetune_{timestamp}")
model = BERTClassifier(learning_rate=1e-4)
trainer = pl.Trainer(default_root_dir="models/bert", max_epochs=2, logger=wandb_logger, accelerator="gpu", devices=[0])
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.save_checkpoint(f"models/bert/bert_model_finetune_{timestamp}.ckpt", weights_only=True)
wandb.finish()

