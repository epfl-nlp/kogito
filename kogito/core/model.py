from abc import ABC, abstractmethod, abstractclassmethod
from kogito.core.knowledge import KnowledgeGraph
from kogito.evaluation.eval import topk_eval


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

    def evaluate(self, input_graph: KnowledgeGraph, top_k=1, *args, **kwargs):
        return evaluate(self, input_graph, top_k=top_k, *args, **kwargs)


def evaluate(model: KnowledgeModel, input_graph: KnowledgeGraph, top_k=1 , *args, **kwargs):
    output_graph = model.generate(input_graph=input_graph, *args, **kwargs)
    evaluation_data = []

    for input_kg, output_kg in zip(input_graph, output_graph):
        assert input_kg.head == output_kg.head
        assert input_kg.relation == output_kg.relation

        evaluation_data.append((output_kg, input_kg.tails))
    
    return topk_eval(evaluation_data, k=top_k)