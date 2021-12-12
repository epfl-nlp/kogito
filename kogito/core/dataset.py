# Importing stock libraries
from pathlib import Path
from typing import Dict
import warnings
import torch
import linecache
import logging
from torch.utils.data import Dataset

from kogito.core.knowledge import GEN_TOKEN, EOS_TOKEN, PAD_TOKEN, ATOMIC_RELATIONS
from kogito.core.utils import encode_line, trim_batch, SortishSampler

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)


class KnowledgeDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        source_len,
        summ_len,
        model="t5",
        is_eval=False,
        head_col: str = "head_event",
        relation_col: str = "relation",
        tail_col: str = "tail_event",
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.data.head_event = (
            self.data[head_col] + " " + self.data[relation_col] + f" {GEN_TOKEN}"
        )
        self.data.tail_event = self.data[tail_col] + f" {EOS_TOKEN}"
        self.text = self.data.head_event
        self.ctext = self.data.tail_event
        self.model = model
        self.is_eval = is_eval
        tokenizer.add_special_tokens(
            {
                "eos_token": EOS_TOKEN,
                "pad_token": PAD_TOKEN,
                "additional_special_tokens": ATOMIC_RELATIONS + [GEN_TOKEN],
            }
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        ctext = str(self.ctext[index])
        ctext = " ".join(ctext.split())

        if self.model == "t5":
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
            if self.is_eval:
                source = self.tokenizer.batch_encode_plus(
                    [text],
                    pad_to_max_length=False,
                    max_length=self.source_len,
                    return_tensors="pt",
                    truncation=True,
                )
                target = self.tokenizer.batch_encode_plus(
                    [ctext],
                    pad_to_max_length=False,
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
        if index < 5:
            logger.info(
                "Source: {}".format(self.tokenizer.batch_decode(source["input_ids"]))
            )
            logger.info(
                "Target: {}".format(self.tokenizer.batch_decode(target["input_ids"]))
            )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_mask.to(dtype=torch.long),
        }


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
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
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n"
        )
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
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
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip(
            "\n"
        )
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
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
