import logging
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
import inspect
import time
import warnings
from collections import defaultdict
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

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
    MBartTokenizer,
)

from kogito.core.utils import (
    assert_all_frozen,
    lmap,
    flatten_list,
    pickle_save,
    save_json,
    freeze_params,
    calculate_rouge,
    ROUGE_KEYS,
    calculate_bleu_score,
)
from kogito.core.dataset import Seq2SeqDataset, MBartDataset
from kogito.models.bart.config import COMETBARTConfig
from kogito.core.callbacks import LoggingCallback

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        config: COMETBARTConfig,
        num_labels=None,
        mode="base",
        pretrained_config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
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

    def setup(self, stage=None):
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
    logger=True,  # can pass WandbLogger() here
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs,
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
        **train_params,
    )

    trainer.fit(model)

    return trainer


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    val_metric = "rouge2"

    def __init__(self, config, train_graph, val_graph, test_graph, **kwargs):
        super().__init__(config, num_labels=None, mode=self.mode, **kwargs)
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.test_graph = test_graph
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.config_save_path = Path(self.output_dir) / "config.pkl"
        pickle_save(self.config, self.config_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        self.dataset_kwargs: dict = dict(
            train_graph=train_graph,
            val_graph=val_graph,
            test_graph=test_graph,
            max_source_length=self.config.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.config.n_train,
            "val": self.config.n_val,
            "test": self.config.n_test,
        }
        self.n_obs = {
            k: v if v >= 0 else None for k, v in n_observations_per_split.items()
        }

        self.target_lens = {
            "train": self.config.max_target_length,
            "val": self.config.val_max_target_length,
            "test": self.config.test_max_target_length,
        }
        assert (
            self.target_lens["train"] <= self.target_lens["val"]
        ), f"target_lens: {self.target_lens}"
        assert (
            self.target_lens["train"] <= self.target_lens["test"]
        ), f"target_lens: {self.target_lens}"

        if self.config.freeze_embeds:
            self.freeze_embeds()
        if self.config.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        try:
            self.num_workers = config.num_workers
        except AttributeError:
            self.num_workers = 2

        self.decoder_start_token_id = None
        self.dataset_class = Seq2SeqDataset

    def freeze_embeds(self):
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["decoder_input_ids"],
        )
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]
        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names
        }
        loss = losses["loss"]
        rouges = {
            k: np.array([x[k] for x in outputs]).mean()
            for k in self.metric_names + ["gen_time", "summ_len"]
        }
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(
            loss
        )
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["avg_rouge1"] = losses["rouge1"]
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {
            "log": metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": rouge_tensor,
        }

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = Seq2SeqDataset.trim_seq2seq_batch(
            batch, pad_token_id
        )
        t0 = time.time()
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        gen_time = (time.time() - t0) / source_ids.shape[0]
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(y)
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time, summ_len=summ_len, preds=preds, target=target, **rouge
        )
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(
        self, type_path: str, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        dataset = self.get_dataset(type_path)
        sampler = None
        if self.config.sortish_sampler and type_path == "train":
            assert self.config.gpus <= 1
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(
            "train", batch_size=self.config.train_batch_size, shuffle=True
        )
        t_total = (
            (
                len(dataloader.dataset)
                // (self.config.train_batch_size * max(1, self.config.gpus))
            )
            // self.config.accumulate_grad_batches
            * float(self.config.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=t_total,
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.config.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.config.eval_batch_size)


class TranslationModule(SummarizationModule):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    val_metric = "bleu"

    def __init__(self, config, train_graph, val_graph, test_graph, **kwargs):
        super().__init__(config, train_graph, val_graph, test_graph, **kwargs)
        self.dataset_kwargs["src_lang"] = config.src_lang
        self.dataset_kwargs["tgt_lang"] = config.tgt_lang
        if self.model.config.decoder_start_token_id is None and isinstance(
            self.tokenizer, MBartTokenizer
        ):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[
                config.tgt_lang
            ]
        if isinstance(self.tokenizer, MBartTokenizer):
            self.dataset_class = MBartDataset

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu_score(preds, target)
