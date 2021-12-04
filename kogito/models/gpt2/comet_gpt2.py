# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from typing import List
import sys

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb
import logging

from torch import cuda

from split.utils import write_items

from optparse import OptionParser

device = 'cuda' if cuda.is_available() else 'cpu'

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.DEBUG)

from kogito.core.modeling import train, validate, beam_generations
from kogito.core.dataset import KnowledgeDataset


class COMET_GPT2(KnowledgeModel):
    def __init__(self, gpt2_model: str = 'gpt2', gpt2_tokenizer: str = 'gpt2-xl'):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model = model.to(device)
    
    def train(self, train_dataset: KnowledgeDataset, val_dataset: KnowledgeDataset, test_dataset: KnowledgeDataset,
              batch_size: int = 2, epochs: int = 3, lr_rate: float = 1e-5, seed: int = 42, callbacks: List[Callback]):
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        torch.backends.cudnn.deterministic = True
        self.tokenizer = train_dataset.tokenizer
        self.model.resize_token_embeddings(len(self.tokenizer))

        train_params = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 0
        }

        val_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0
        }

        train_loader = DataLoader(train_dataset, **train_params, drop_last=True)
        val_loader = DataLoader(val_dataset, **val_params, drop_last=True)
        test_loader = DataLoader(test_dataset, **val_params, drop_last=True)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_rate)

        for epoch in range(config.epochs):
            train(epoch, self.tokenizer, model, device, train_loader, optimizer, val_loader, model_class="gpt2")
            for callback in callbacks:
                callback(self)
        
        return self.model

    
    def generate(self, dataset: KnowledgeDataset, top_k: int = 1):
        params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0
        }
        loader = DataLoader(dataset, **params, drop_last=False)
        generations = beam_generations(self.tokenizer, self.model, device, loader, top_k=top_k)
        return generations
