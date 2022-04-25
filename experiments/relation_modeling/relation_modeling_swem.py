import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import wandb

from relation_modeling_utils import SWEMHeadDataset, load_fdata, load_data, MaxPool, AvgPool, Evaluator, get_timestamp

DATASET_TYPE = "n5"
NUM_EPOCHS = 20
LR_RATE = 1e-4
FREEZE_EMB = False
BATCH_SIZE = 128
POOLING = "avg"

VOCAB, EMBEDDING_MATRIX = np.load("data/vocab_glove_100d.npy", allow_pickle=True).item(), np.load("data/embedding_matrix_glove_100d.npy", allow_pickle=True)

class SWEMClassifier(Evaluator, pl.LightningModule):
    def __init__(self, num_classes=3, pooling="avg", freeze_emb=False, learning_rate=1e-4):
        super(SWEMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=EMBEDDING_MATRIX.shape[0],
                                      embedding_dim=EMBEDDING_MATRIX.shape[1]).from_pretrained(torch.tensor(EMBEDDING_MATRIX, dtype=torch.float32), freeze=freeze_emb)
        self.pool = MaxPool() if pooling == "max" else AvgPool()
        self.linear = nn.Linear(EMBEDDING_MATRIX.shape[1], num_classes)
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



if __name__ == "__main__":
    train_df = load_fdata(f"data/atomic_ood2/{DATASET_TYPE}/train_{DATASET_TYPE}.csv")
    val_df = load_data("data/atomic2020_data-feb2021/dev.tsv", multi_label=True)
    test_df = load_fdata(f"data/atomic_ood2/{DATASET_TYPE}/test_{DATASET_TYPE}.csv")
    train_data = SWEMHeadDataset(train_df, vocab=VOCAB)
    val_data = SWEMHeadDataset(val_df, vocab=VOCAB)
    test_data = SWEMHeadDataset(test_df, vocab=VOCAB)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    timestamp = get_timestamp()

    emb_txt = 'frozen' if FREEZE_EMB else 'finetune'

    wandb_logger = WandbLogger(project="kogito-relation-matcher", name=f"swem_{emb_txt}_{DATASET_TYPE}")
    wandb_logger.experiment.config["epochs"] = NUM_EPOCHS
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE
    model = SWEMClassifier(pooling=POOLING, freeze_emb=FREEZE_EMB, learning_rate=LR_RATE)
    trainer = pl.Trainer(default_root_dir="models/swem", max_epochs=NUM_EPOCHS, logger=wandb_logger, accelerator="gpu", devices=[0])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    trainer.save_checkpoint(f"models/swem/swem_{emb_txt}_{DATASET_TYPE}_{timestamp}.ckpt", weights_only=True)
    wandb.finish()