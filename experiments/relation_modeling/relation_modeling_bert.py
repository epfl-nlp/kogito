from torch.utils.data import DataLoader
import pytorch_lightning as pl
from relation_modeling_utils import load_fdata, get_timestamp, load_data
from pytorch_lightning.loggers import WandbLogger
import wandb

from kogito.core.processors.models.bert import BERTConfig, BERTHeadDataset, BERTClassifier

MODEL_TYPE = "uncased"
NUM_EPOCHS = 3
BATCH_SIZE = 2
FREEZE_EMB = True
DATASET_TYPE = "n1"
LR_RATE = 1e-4

if __name__ == "__main__":
    train_df = load_fdata(f"data/atomic_ood2/{DATASET_TYPE}/train_{DATASET_TYPE}.csv")
    val_df = load_data("data/atomic2020_data-feb2021/dev.tsv", multi_label=True)
    test_df = load_fdata(f"data/atomic_ood2/{DATASET_TYPE}/test_{DATASET_TYPE}.csv")
    train_data = BERTHeadDataset(train_df, tokenizer_type=MODEL_TYPE)
    val_data = BERTHeadDataset(val_df, tokenizer_type=MODEL_TYPE)
    test_data = BERTHeadDataset(test_df, tokenizer_type=MODEL_TYPE)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    timestamp = get_timestamp()
    emb_txt = 'frozen' if FREEZE_EMB else 'finetune'

    wandb_logger = WandbLogger(project="kogito-relation-matcher", name=f"bert_{emb_txt}_{MODEL_TYPE}_{DATASET_TYPE}")
    wandb_logger.experiment.config["epochs"] = NUM_EPOCHS
    wandb_logger.experiment.config["batch_size"] = BATCH_SIZE
    config = BERTConfig(learning_rate=LR_RATE, model_case=MODEL_TYPE, freeze_emb=FREEZE_EMB)
    model = BERTClassifier(config)
    trainer = pl.Trainer(default_root_dir="models/bert", max_epochs=NUM_EPOCHS, logger=wandb_logger, accelerator="gpu", devices=[0])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    trainer.save_checkpoint(f"models/bert/bert_model_{emb_txt}_{MODEL_TYPE}_{DATASET_TYPE}_{timestamp}.ckpt", weights_only=True)
    model.save_pretrained("hmodels/bert")
    wandb.finish()

