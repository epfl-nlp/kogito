import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

import logging

from kogito.core.modeling import train, beam_generations
from kogito.core.dataset import KnowledgeDataset
from kogito.models.base import KnowledgeModel

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)

device = "cuda" if cuda.is_available() else "cpu"


class COMETGPT2(KnowledgeModel):
    def __init__(self, gpt2_model: str = "gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model)

    def train(
        self,
        train_dataset: KnowledgeDataset,
        val_dataset: KnowledgeDataset,
        batch_size: int = 2,
        epochs: int = 3,
        lr_rate: float = 1e-5,
        seed: int = 42,
        callbacks=None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        self.model = self.model.to(device)
        self.tokenizer = train_dataset.tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

        train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}

        val_params = {"batch_size": 1, "shuffle": False, "num_workers": 0}

        train_loader = DataLoader(train_dataset, **train_params, drop_last=True)
        val_loader = DataLoader(val_dataset, **val_params, drop_last=True)

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr_rate)

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
            )
            for callback in callbacks:
                callback(self)

        return self.model

    def generate(self, dataset: KnowledgeDataset, top_k: int = 1):
        params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
        loader = DataLoader(dataset, **params, drop_last=False)
        generations = beam_generations(
            self.tokenizer, self.model, device, loader, top_k=top_k
        )
        return generations
