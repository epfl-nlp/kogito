import logging
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict
from kogito.models.bart.config import COMETBARTConfig
from kogito.core.callbacks import LoggingCallback
import inspect

import pytorch_lightning as pl

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


logger = logging.getLogger(__name__)


MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        config: COMETBARTConfig,
        num_labels=None,
        mode="base",
        pretrained_config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and pretrained_config."""
        super().__init__()
        self.config = config
        self.step_count = 0
        self.tfmr_ckpts = {}
        self.output_dir = Path(self.config.output_dir)
        cache_dir = self.config.cache_dir if self.config.cache_dir else None
        if pretrained_config is None:
            self.pretrained_config = AutoConfig.from_pretrained(
                self.config.pretrained_config
                if self.config.pretrained_config
                else self.config.pretrained_model,
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.pretrained_config: PretrainedConfig = pretrained_config
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.pretrained_tokenizer
                if self.config.pretrained_tokenizer
                else self.config.pretrained_model,
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.config.pretrained_model,
                from_tf=bool(".ckpt" in self.config.pretrained_model),
                config=self.pretrained_config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )
        self.opt = optimizer

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def setup(self, step):
        train_batch_size = self.config.train_batch_size
        dataloader = self.get_dataloader("train", train_batch_size)
        self.train_loader = dataloader
        self.total_steps = (
            (len(dataloader.dataset) // (train_batch_size * max(1, self.config.gpus)))
            // self.config.accumulate_grad_batches
            * float(self.config.num_train_epochs)
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.config.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test", self.config.eval_batch_size)

    def _feature_file(self, mode):
        return os.path.join(
            self.config.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.config.pretrained_model.split("/"))).pop(),
                str(self.config.max_seq_length),
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        save_path.mkdir(exist_ok=True)
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.tfmr_ckpts[self.step_count] = save_path


def generic_train(
    model: BaseTransformer,
    config: COMETBARTConfig,
    early_stopping_callback=False,
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    pl.seed_everything(config.seed)

    # init model
    odir = Path(model.config.output_dir)
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=config.output_dir,
            prefix="checkpoint",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if config.fp16:
        train_params["precision"] = 16
        train_params["amp_level"] = config.fp16_opt_level

    if config.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer_param_keys = inspect.signature(pl.Trainer).parameters.keys()

    trainer = pl.Trainer(
        **{
            ckey: cval
            for ckey, cval in asdict(config).items()
            if ckey in trainer_param_keys
        },
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
        **train_params,
    )

    trainer.fit(model)

    return trainer
