from abc import ABC, abstractmethod, abstractclassmethod

class KnowledgeModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, filepath: str):
        raise NotImplementedError
    
    @abstractclassmethod
    def load(self, filepath: str):
        raise NotImplementedError