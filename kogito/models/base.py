from abc import ABC, abstractmethod, abstractclassmethod


class KnowledgeModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, save_dir: str):
        raise NotImplementedError

    @abstractclassmethod
    def from_pretrained(cls, model_name_or_path: str):
        raise NotImplementedError
