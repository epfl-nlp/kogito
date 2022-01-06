import numpy as np
import torch
from torch import cuda
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from kogito.core.utils import find_nth
from kogito.models.base import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph

device = "cuda" if cuda.is_available() else "cpu"


class GPT2Zeroshot(KnowledgeModel):
    def __init__(self, gpt2_model: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.model.to(device)

    def train(self):
        raise ValueError("GPT-2 Zeroshot model is not trainable")

    def save(self, save_model_path):
        if save_model_path:
            self.model.save_pretrained(save_model_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        return cls(model_name_or_path)

    def generate(
        self,
        input_graph: KnowledgeGraph,
        seed: int = 42,
        top_k: int = 1,
        top_p: float = 0.9,
        num_sequences: int = 10,
        num_beams: int = 10,
        stop_token: str = ".",
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        outputs = []
        for input_kg in input_graph:
            prompt = input_kg.to_prompt()
            input_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            generations = self.model.generate(
                input_ids=input_ids.to(device),
                max_length=input_ids.size(1) + 10,
                temperature=1.0,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=num_sequences,
                num_beams=num_beams,
            )

            if len(generations.shape) > 2:
                generations.squeeze_()

            text_generations = []
            for gen in generations:
                gen = gen.tolist()
                text = self.tokenizer.decode(gen, clean_up_tokenization_spaces=True)
                text = (
                    text[: find_nth(text, stop_token, 1)]
                    if stop_token not in prompt
                    else text[: find_nth(text, stop_token, 2)]
                )
                text_generations.append(text)

            output_kg = input_kg.copy()
            output_kg.tails = text_generations
            outputs.append(output_kg)

        return KnowledgeGraph(outputs)
