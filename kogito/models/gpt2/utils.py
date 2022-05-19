import torch
import pytorch_lightning as pl


class GPT2Finetuner(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-5) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, input_ids, mask):
        return self.model(input_ids=input_ids, attention_mask=mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        X = batch
        ids = X["source_ids"]
        mask = X["source_mask"]
        outputs = self.model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch
        ids = X["source_ids"]
        mask = X["source_mask"]
        outputs = self.model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
