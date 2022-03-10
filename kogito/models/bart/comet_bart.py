import glob
import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers import MBartTokenizer, get_linear_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from kogito.core.knowledge import (
    KG_RELATIONS,
    KnowledgeGraph,
)
from kogito.core.utils import (
    assert_all_frozen,
    use_task_specific_params,
    lmap,
    flatten_list,
    pickle_save,
    save_json,
    freeze_params,
    calculate_rouge,
    ROUGE_KEYS,
    calculate_bleu_score,
    chunks,
    trim_batch,
)
from kogito.core.dataset import Seq2SeqDataset, MBartDataset
from kogito.core.callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback
from kogito.models.base import KnowledgeModel
from kogito.models.bart.config import COMETBARTConfig
from kogito.models.bart.lightning import BaseTransformer, generic_train

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


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
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
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


class COMETBART(KnowledgeModel):
    def __init__(self, config, **kwargs) -> None:
        self.model = None
        self.tokenizer = None
        self.config = config
        self.kwargs = kwargs

    def train(
        self,
        train_graph: KnowledgeGraph,
        val_graph: KnowledgeGraph,
        test_graph: KnowledgeGraph,
        logger_name: str = "default",
    ):
        Path(self.config.output_dir).mkdir(exist_ok=True)

        if self.config.task == "summarization":
            self.model: SummarizationModule = SummarizationModule(
                self.config, train_graph, val_graph, test_graph, **self.kwargs
            )
        elif self.config.task == "translation":
            self.model: SummarizationModule = TranslationModule(
                self.config, train_graph, val_graph, test_graph, **self.kwargs
            )
        else:
            raise ValueError

        if self.config.atomic:
            self.model.tokenizer.add_tokens(KG_RELATIONS)
            self.model.model.resize_token_embeddings(len(self.model.tokenizer))

        if (
            logger_name == "default"
            or str(self.config.output_dir).startswith("/tmp")
            or str(self.config.output_dir).startswith("/var")
        ):
            logger = True  # don't pollute wandb logs unnecessarily

        elif logger_name == "wandb":
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(
                name=self.model.output_dir.name, project=self.model.output_dir.name
            )

        elif logger_name == "wandb_shared":
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(
                name=self.model.output_dir.name,
                project=f"hf_{self.model.output_dir.name}",
            )

        trainer: pl.Trainer = generic_train(
            self.model,
            self.config,
            logging_callback=Seq2SeqLoggingCallback(),
            checkpoint_callback=get_checkpoint_callback(
                self.config.output_dir, self.model.val_metric
            ),
            logger=logger,
        )
        pickle_save(self.model.config, self.model.output_dir / "config.pkl")

        self.model.config.test_checkpoint = ""
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(self.config.output_dir, "*.ckpt"), recursive=True
                )
            )
        )
        if checkpoints:
            self.model.config.test_checkpoint = checkpoints[-1]
            trainer.resume_from_checkpoint = checkpoints[-1]
        trainer.logger.log_hyperparams(asdict(self.model.config))

        trainer.test(self.model)

    def generate(
        self,
        input_graph: KnowledgeGraph,
        decode_method="beam",
        num_generate=3,
        batch_size=64,
    ):
        with torch.no_grad():
            outputs = []
            for kg_batch in list(chunks(input_graph, batch_size)):
                queries = []
                for kg_input in kg_batch:
                    queries.append(kg_input.to_query(decode_method=decode_method))
                batch = self.tokenizer(
                    queries, return_tensors="pt", truncation=True, padding="max_length"
                ).to(device)
                input_ids, attention_mask = trim_batch(
                    **batch, pad_token_id=self.tokenizer.pad_token_id
                )

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.config.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                )

                output = self.tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for kg_input, generations in zip(kg_batch, list(chunks(output, num_generate))):
                    output_kg = kg_input.copy()
                    output_kg.tails = generations
                    outputs.append(output_kg)

            return KnowledgeGraph(outputs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, task: str = "summarization"):
        config = COMETBARTConfig(task=task, decoder_start_token_id=None)
        comet_bart = cls(config)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        use_task_specific_params(model, task)
        comet_bart.model = model
        comet_bart.tokenizer = tokenizer
        model.to(device)
        return comet_bart

    def save(self, filepath):
        pass
