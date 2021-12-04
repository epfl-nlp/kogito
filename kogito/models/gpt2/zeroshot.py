# Importing stock libraries
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from kogito.core.utils import find_nth

class GPT2Zeroshot(KnowledgeModel):
    def __init__(self, gpt2_model: str = 'gpt2', save_model_path: Optional[str] = None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(device)
        if save_model_path:
            model.save_pretrained(save_model_path)
    
    def train(self):
        raise ValueError('GPT-2 Zeroshot model is not trainable')
    
    def generate(self, inputs: List[Knowledge], seed: int = 42, top_k: = 1, top_p: float = 0.9,
                 num_sequences: int = 10, num_beams: int = 10, stop_token: str = '.'):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        outputs = []
        for kg_input in inputs:
            prompt = kg_input.to_prompt()
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            generations = self.model.generate(
                input_ids=input_ids.to(device),
                max_length=input_ids.size(1) + 10,
                temperature=1.0,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=num_sequences,
                num_beams=num_beams
            )

            if len(generations.shape) > 2:
                generations.squeeze_()

            text_generations = []
            for gen in generations:
                gen = gen.tolist()
                text = tokenizer.decode(gen, clean_up_tokenization_spaces=True)
                text = text[:find_nth(text, stop_token, 1)] if stop_token not in prompt else text[:find_nth(text, stop_token, 2)]
                text_generations.append(text)

            outputs.append(Knowledge(base=kg_input.base, head=kg_input.head, relation=kg_input.relation, tails=text_generations))
        
        return outputs