import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wandb
import logging

from kogito.core.modeling import train, beam_generations
from kogito.core.dataset import KnowledgeDataset
from kogito.models.base import KnowledgeModel
from kogito.core.knowledge import (
    KnowledgeGraph,
    GEN_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    KG_RELATIONS,
)

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)

device = "cuda" if cuda.is_available() else "cpu"


class COMETGPT2(KnowledgeModel):
    def __init__(self, model_name_or_path: str = "gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model.to(device)

    def train(
        self,
        train_graph: KnowledgeGraph,
        val_graph: KnowledgeGraph,
        batch_size: int = 2,
        in_len: int = 16,
        out_len: int = 34,
        summary_len: int = 0,
        epochs: int = 3,
        lr_rate: float = 1e-5,
        seed: int = 42,
        log_wandb: bool = False,
        output_dir=None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        self.tokenizer.add_special_tokens(
            {
                "eos_token": EOS_TOKEN,
                "pad_token": PAD_TOKEN,
                "additional_special_tokens": KG_RELATIONS + [GEN_TOKEN],
            }
        )

        train_dataset = KnowledgeDataset(
            train_graph,
            tokenizer=self.tokenizer,
            source_len=out_len,
            summ_len=summary_len,
            model="gpt2",
        )
        val_dataset = KnowledgeDataset(
            val_graph,
            tokenizer=self.tokenizer,
            source_len=in_len,
            summ_len=out_len - in_len,
            model="gpt2",
            is_eval=True,
        )
        train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
        val_params = {"batch_size": 1, "shuffle": False, "num_workers": 0}

        train_loader = DataLoader(train_dataset, **train_params, drop_last=True)
        val_loader = DataLoader(val_dataset, **val_params, drop_last=True)

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr_rate)

        self.model.resize_token_embeddings(len(self.tokenizer))

        if log_wandb:
            config = {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr_rate,
                "seed": seed,
                "in_len": in_len,
                "summary_len": summary_len,
                "out_len": out_len,
            }
            wandb.init(project="kogito_comet_gpt2", config=config)

        for epoch in range(epochs):
            train(
                epoch,
                self.tokenizer,
                self.model,
                device,
                train_loader,
                optimizer,
                val_loader,
                model_class="gpt2",
                log_wandb=log_wandb,
                output_dir=output_dir,
            )
            if output_dir:
                self.model.save_pretrained("{}/checkpoint_{}".format(output_dir, epoch))
                self.tokenizer.save_pretrained(
                    "{}/checkpoint_{}".format(output_dir, epoch)
                )

        return self.model

    def generate(
        self,
        input_graph: KnowledgeGraph,
        in_len: int = 16,
        out_len: int = 34,
        top_k: int = 1,
    ):
        params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
        dataset = KnowledgeDataset(
            input_graph,
            tokenizer=self.tokenizer,
            source_len=in_len,
            summ_len=out_len - in_len,
            model="gpt2",
            is_eval=True,
        )
        loader = DataLoader(dataset, **params, drop_last=False)
        generations = beam_generations(
            self.tokenizer, self.model, device, loader, top_k=top_k
        )
        outputs = []
        for input_kg, gen in zip(input_graph, generations):
            output_kg = input_kg.copy()
            output_kg.tails = gen["generations"]
            outputs.append(output_kg)

        return KnowledgeGraph(outputs)

    def save(self, filepath):
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        return cls(model_name_or_path)
