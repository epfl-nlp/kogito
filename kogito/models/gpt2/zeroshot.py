import numpy as np
import torch
from torch import cuda
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from kogito.core.utils import find_nth
from kogito.core.model import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph

device = "cuda" if cuda.is_available() else "cpu"


class GPT2Zeroshot(KnowledgeModel):
    """Zeroshot knowledge model based on GPT-2"""

    def __init__(self, gpt2_model: str = "gpt2") -> None:
        """Initialize GPT-2 model

        Args:
            gpt2_model (str, optional): HuggingFace model name for gpt2. Defaults to "gpt2".
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.model.to(device)

    def train(self):
        raise ValueError("GPT-2 Zeroshot model is not trainable")

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = "gpt2"):
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
    ) -> KnowledgeGraph:
        """Generate inferences from GPT2 model

        Args:
            input_graph (KnowledgeGraph): Input dataset
            seed (int, optional): Random seed. Defaults to 42.
            top_k (int, optional): Top k. Defaults to 1.
            top_p (float, optional): Top p. Defaults to 0.9.
            num_sequences (int, optional): Number of sequences. Defaults to 10.
            num_beams (int, optional): Number of beams. Defaults to 10.
            stop_token (str, optional): Stop token. Defaults to ".".

        Returns:
            KnowledgeGraph: Completed knowledge graph
        """
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
