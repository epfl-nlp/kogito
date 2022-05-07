# Importing stock libraries
from pathlib import Path
from typing import Dict
import warnings
import torch
from torch.utils.data import Dataset

from kogito.core.knowledge import GEN_TOKEN, EOS_TOKEN
from kogito.core.utils import encode_line, trim_batch, SortishSampler


class KnowledgeDataset(Dataset):
    def __init__(self, kg_graph, tokenizer, source_len, summ_len, is_eval=False):
        self.tokenizer = tokenizer
        self.data = kg_graph.to_dataframe()
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data["head"] + " " + self.data["relation"] + f" {GEN_TOKEN}"
        self.ctext = self.data["tails"] + f" {EOS_TOKEN}"
        self.is_eval = is_eval

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        ctext = str(self.ctext[index])
        ctext = " ".join(ctext.split())

        if self.is_eval:
            source = self.tokenizer.batch_encode_plus(
                [text],
                pad_to_max_length=True,
                max_length=self.source_len,
                return_tensors="pt",
                truncation=True,
            )
            target = self.tokenizer.batch_encode_plus(
                [ctext],
                pad_to_max_length=True,
                max_length=self.summ_len,
                return_tensors="pt",
                truncation=True,
            )
        else:
            source = self.tokenizer.batch_encode_plus(
                [text + " " + ctext],
                pad_to_max_length=True,
                max_length=self.source_len + self.summ_len,
                return_tensors="pt",
                truncation=True,
            )
            target = source

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        train_graph,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        val_graph=None,
        test_graph=None,
    ):
        super().__init__()
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.test_graph = test_graph
        self.input_graph = self.train_graph

        if type_path == "val":
            self.input_graph = self.val_graph

        if type_path == "test":
            self.input_graph = self.test_graph

        self.src_lens = [
            len(f"{kg.head} {kg.relation} {GEN_TOKEN}") for kg in self.input_graph
        ]
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        kg = self.input_graph[index]
        source_line = self.prefix + f"{kg.head} {kg.relation} {GEN_TOKEN}"
        tgt_line = kg.tails[0] if kg.tails else None
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(
            batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
        )
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(
            input_ids, pad_token_id, attention_mask=masks
        )
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)


class MBartDataset(Seq2SeqDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_source_length != self.max_target_length:
            warnings.warn(
                f"Mbart will ignore max_target_length = {self.max_target_length} and"
                "use {self.max_source_length} for both sides."
            )

    def __getitem__(self, index) -> Dict[str, str]:
        kg = self.input_graph[index]
        source_line = self.prefix + f"{kg.head} {kg.relation} {GEN_TOKEN}"
        tgt_line = kg.tails[0] if kg.tails else None
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {
            "tgt_texts": source_line,
            "src_texts": tgt_line,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_translation_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
        )
        return batch_encoding.data
