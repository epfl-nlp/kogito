import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from kogito.core.processors.models.swem import SWEMConfig, SWEMClassifier, SWEMHeadDataset
from relation_modeling_utils import load_fdata, load_data, get_timestamp

DATASET_TYPE = "n1"
NUM_EPOCHS = 20
LR_RATE = 1e-4
FREEZE_EMB = False
BATCH_SIZE = 128
POOLING = "avg"

VOCAB = np.load("data/vocab_glove_100d.npy", allow_pickle=True).item()

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
    config = SWEMConfig(pooling=POOLING, freeze_emb=FREEZE_EMB, learning_rate=LR_RATE)
    model = SWEMClassifier(config)
    trainer = pl.Trainer(default_root_dir="models/swem", max_epochs=NUM_EPOCHS, logger=wandb_logger, accelerator="gpu", devices=[0])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    trainer.save_checkpoint(f"models/swem/swem_{emb_txt}_{DATASET_TYPE}_{timestamp}.ckpt", weights_only=True)
    model.save_pretrained("hmodels/swem")
    wandb.finish()