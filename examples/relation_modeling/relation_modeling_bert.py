import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from relation_modeling_utils import load_fdata, get_timestamp
from pytorch_lightning.loggers import WandbLogger
import wandb


MODEL_TYPE = "uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 2
FREEZE_EMB = True
DATASET_TYPE = "f1"

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


class BERTClassifier(pl.LightningModule):
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
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision(num_classes=3, average='weighted')
        self.val_precision = torchmetrics.Precision(num_classes=3, average='weighted')
        self.train_recall = torchmetrics.Recall(num_classes=3, average='weighted')
        self.val_recall = torchmetrics.Recall(num_classes=3, average='weighted')
        self.train_f1 = torchmetrics.F1Score(num_classes=3, average='weighted')
        self.val_f1 = torchmetrics.F1Score(num_classes=3, average='weighted')

        self.test_accuracy = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision(num_classes=3, average='weighted')
        self.test_recall = torchmetrics.Recall(num_classes=3, average='weighted')
        self.test_f1 = torchmetrics.F1Score(num_classes=3, average='weighted')

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

    def test_step(self, batch, batch_idx):
        X, y = batch
        mask = X['attention_mask']
        input_ids = X['input_ids'].squeeze(1)
        outputs = self.forward(input_ids, mask)
        test_loss = self.criterion(outputs, y.float())
        preds = F.sigmoid(outputs)
        self.test_accuracy(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        self.log("test_loss", test_loss, on_epoch=True)
        self.log('test_accuracy', self.test_accuracy, on_epoch=True)
        self.log('test_precision', self.test_precision, on_epoch=True)
        self.log('test_recall', self.test_recall, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



if __name__ == "__main__":
    train_df = load_fdata(f"data/atomic_ood/{DATASET_TYPE}/train_{DATASET_TYPE}.csv")
    val_df = load_fdata(f"data/atomic_ood/{DATASET_TYPE}/val_{DATASET_TYPE}.csv")
    test_df = load_fdata(f"data/atomic_ood/{DATASET_TYPE}/test_{DATASET_TYPE}.csv")
    train_data = BERTHeadDataset(train_df, tokenizer_type=MODEL_TYPE)
    val_data = BERTHeadDataset(val_df, tokenizer_type=MODEL_TYPE)
    test_data = BERTHeadDataset(test_df, tokenizer_type=MODEL_TYPE)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    timestamp = get_timestamp()
    emb_txt = 'frozen' if FREEZE_EMB else 'finetune'

    wandb_logger = WandbLogger(project="kogito-relation-matcher", name=f"bert_{emb_txt}_{MODEL_TYPE}_{DATASET_TYPE}")
    wandb_logger.experiment.config["epochs"] = NUM_EPOCHS
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE
    model = BERTClassifier(learning_rate=1e-4, model_type=MODEL_TYPE, freeze_emb=FREEZE_EMB)
    trainer = pl.Trainer(default_root_dir="models/bert", max_epochs=NUM_EPOCHS, logger=wandb_logger, accelerator="gpu", devices=[0])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    trainer.save_checkpoint(f"models/bert/bert_model_{emb_txt}_{MODEL_TYPE}_{DATASET_TYPE}_{timestamp}.ckpt", weights_only=True)
    wandb.finish()

