import time

import openai

from kogito.models.base import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph


class GPT3Zeroshot(KnowledgeModel):
    def __init__(self, api_key: str, model_name: str = "text-davinci-002"):
        self.api_key = api_key
        self.model_name = model_name

    def train(self):
        raise ValueError("GPT-3 Zeroshot model is not trainable")

    def save(self, save_model_path):
        raise ValueError("GPT-3 Zeroshot cannot be saved")

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        raise ValueError("GPT-3 only supports API-based access")

    def generate(
        self,
        input_graph: KnowledgeGraph,
        num_samples: int = 5,
        max_tokens: int = 16,
        temperature: float = 0.9,
        top_p: float = 1,
        n: int = 1,
        logprobs: int = None,
        stop: str = "\n"
    ):
        rel_kg_map = {}
        outputs = []

        for input_kg in input_graph:
            if input_kg.relation not in rel_kg_map:
                rel_kg_map[input_kg.relation] = {'samples': [], 'targets': []}
            if input_kg.tails:
                rel_kg_map[input_kg.relation]['samples'].append(input_kg)
            else:
                rel_kg_map[input_kg.relation]['targets'].append(input_kg)
        
        for relation, kg_map in rel_kg_map.items():
            samples = kg_map['samples'][:num_samples]
            targets = kg_map['targets']
            prompts = []
            sample_prompt = '\n'.join([sample_kg.to_prompt(include_tail=True) for sample_kg in samples])
            
            for target in targets:
                prompts.append(f"{sample_prompt}\n{target.to_prompt()}")

            responses = complete_gpt3(api_key=self.api_key, model_name=self.model_name, prompt=prompts,
                                     max_tokens=max_tokens, temperature=temperature, top_p=top_p, logprobs=logprobs, n=n, stop=stop)
            
            for target, response in zip(targets, responses):
                output_kg = target.copy()
                output_kg.tails = [choice["text"] for choice in response["choices"]]
                outputs.append(output_kg)

        return KnowledgeGraph(outputs)


def complete_gpt3(api_key, model_name, prompt, max_tokens=16, temperature=1, top_p=1, logprobs=None, n=1, stop="\n"):
    response = None
    openai.api_key = api_key

    try:
        response = openai.Completion.create(engine=model_name, 
                                            prompt=prompt,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            logprobs=logprobs,
                                            echo=False,
                                            stop=stop,
                                            n=n)
        received = True
    except Exception as e:
        print("Something went wrong when querying GPT-3 API")
        raise e
    return response
