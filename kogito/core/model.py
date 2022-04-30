from abc import ABC, abstractmethod, abstractclassmethod
from kogito.core.knowledge import KnowledgeGraph


class KnowledgeModel(ABC):
    """
    Base class to represent a Knowledge Model.
    """

    @abstractmethod
    def train(self, train_graph: KnowledgeGraph, *args, **kwargs) -> "KnowledgeModel":
        """Train a knowledge model

        Args:
            train_graph (KnowledgeGraph): Training dataset

        Raises:
            NotImplementedError: This method has to be implemented
                                 by concrete subclasses.

        Returns:
            KnowledgeModel: Trained knowledge model
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, input_graph: KnowledgeGraph, *args, **kwargs) -> KnowledgeGraph:
        """Generate inferences from knowledge model

        Args:
            input_graph (KnowledgeGraph): Input dataset

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses.

        Returns:
            KnowledgeGraph: Input graph with tails generated
        """
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(self, save_path: str) -> None:
        """Save model as a pretrained model

        Args:
            save_path (str): Directory to save the model to.

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses.
        """
        raise NotImplementedError

    @abstractclassmethod
    def from_pretrained(cls, model_name_or_path: str) -> "KnowledgeModel":
        """Load model from a pretrained model path
        This method can load models either from HuggingFace by model name
        or from disk by model path.

        Args:
            model_name_or_path (str): HuggingFace model name or local model path to load from.

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses.

        Returns:
            KnowledgeModel: Loaded knowledge model.
        """
        raise NotImplementedError
