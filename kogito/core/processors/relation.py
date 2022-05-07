from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Type
from functools import partial
import pkgutil
from io import BytesIO

import numpy as np
import torch
import pytorch_lightning as pl
from spacy.language import Language
from torch.utils.data import DataLoader, Dataset

from kogito.core.head import KnowledgeHead
from kogito.core.relation import (
    KnowledgeRelation,
    HEAD_TO_RELATION_MAP,
    PHYSICAL_RELATIONS,
    EVENT_RELATIONS,
    SOCIAL_RELATIONS,
)

from kogito.core.processors.models.swem import SWEMHeadDataset, SWEMClassifier
from kogito.core.processors.models.distilbert import (
    DistilBERTHeadDataset,
    DistilBERTClassifier,
)
from kogito.core.processors.models.bert import BERTHeadDataset, BERTClassifier

RELATION_CLASSES = [PHYSICAL_RELATIONS, EVENT_RELATIONS, SOCIAL_RELATIONS]


class KnowledgeRelationMatcher(ABC):
    """Base class for relation matching"""

    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        """Initialize relation matcher

        Args:
            name (str): Unique relation matcher name
            lang (Optional[Language], optional): Spacy language pipeline to use. Defaults to None.
        """
        self.name = name
        self.lang = lang

    @abstractmethod
    def match(
        self,
        heads: List[KnowledgeHead],
        relations: Optional[List[KnowledgeRelation]] = None,
        **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        """Match relations to given heads

        Args:
            heads (List[KnowledgeHead]): List of heads to match for.
            relations (Optional[List[KnowledgeRelation]], optional): Subset of relations to use for matching.
                                                                    Defaults to None.

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses

        Returns:
            List[Tuple[KnowledgeHead, KnowledgeRelation]]: List of matched head, relation tuples
        """
        raise NotImplementedError


class SimpleRelationMatcher(KnowledgeRelationMatcher):
    """Matches relation based on simple heuristics"""

    def match(
        self,
        heads: List[KnowledgeHead],
        relations: List[KnowledgeRelation] = None,
        **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        head_relations = []

        for head in heads:
            rels_to_match = HEAD_TO_RELATION_MAP[head.type]
            if relations:
                rels_to_match = set(HEAD_TO_RELATION_MAP[head.type]).intersection(
                    set(relations)
                )
            for relation in rels_to_match:
                head_relations.append((head, relation))

        return head_relations


class ModelBasedRelationMatcher(KnowledgeRelationMatcher):
    """Matches relations based on relation classifiers"""

    def __init__(
        self,
        name: str,
        dataset_class: Type[Dataset],
        model_class: Type[pl.LightningModule],
        model_path: str,
        batch_size: int = 64,
        lang: Optional[Language] = None,
    ) -> None:
        """Initialize a model based relation matcher

        Args:
            name (str): Unique relation matcher name
            dataset_class (Type[Dataset]): Dataset class to use
            model_class (Type[pl.LightningModule]): Model class to use
            model_path (str): Model path to load model from
            batch_size (int, optional): Batch size for inference. Defaults to 64.
            lang (Optional[Language], optional): Spacy lang pipeline. Defaults to None.
        """
        super().__init__(name, lang)
        self.dataset_class = dataset_class
        self.model_class = model_class
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = model_class.from_pretrained(model_path)

    def match(
        self,
        heads: List[KnowledgeHead],
        relations: List[KnowledgeRelation] = None,
        **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        data = [str(head) for head in heads]
        dataset = self.dataset_class(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        trainer = pl.Trainer()
        predictions = torch.cat(
            trainer.predict(self.model, dataloaders=dataloader)
        ).numpy()
        head_relations = []

        for head, prob in zip(heads, predictions):
            prediction = np.where(prob >= 0.5, 1, 0).tolist()
            pred_rel_classes = [
                RELATION_CLASSES[idx]
                for idx, pred in enumerate(prediction)
                if pred == 1
            ]

            if not pred_rel_classes:
                pred_rel_classes = [RELATION_CLASSES[np.argmax(prob)]]

            for rel_class in pred_rel_classes:
                rels_to_match = rel_class
                if relations:
                    rels_to_match = set(rels_to_match).intersection(set(relations))
                for relation in rels_to_match:
                    head_relations.append((head, relation))

        return head_relations


class SWEMRelationMatcher(ModelBasedRelationMatcher):
    """Relation matcher based on Simple Word Embeddings (GloVes)"""

    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        vocab = np.load(
            BytesIO(pkgutil.get_data(__name__, "data/vocab_glove_100d.npy")),
            allow_pickle=True,
        ).item()
        dataset_class = partial(SWEMHeadDataset, vocab=vocab, lang=lang)
        model_class = SWEMClassifier
        model_path = "mismayil/kogito-rc-swem"
        super().__init__(
            name,
            dataset_class=dataset_class,
            model_class=model_class,
            model_path=model_path,
            lang=lang,
        )


class DistilBERTRelationMatcher(ModelBasedRelationMatcher):
    """Relation matcher based on DistilBERT embeddings"""

    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        dataset_class = DistilBERTHeadDataset
        model_class = DistilBERTClassifier
        model_path = "mismayil/kogito-rc-distilbert"
        super().__init__(
            name,
            dataset_class=dataset_class,
            model_class=model_class,
            model_path=model_path,
            lang=lang,
        )


class BERTRelationMatcher(ModelBasedRelationMatcher):
    """Relation matcher based on BERT embeddings"""

    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        dataset_class = BERTHeadDataset
        model_class = BERTClassifier
        model_path = "mismayil/kogito-rc-bert"
        super().__init__(
            name,
            dataset_class=dataset_class,
            model_class=model_class,
            model_path=model_path,
            lang=lang,
        )


class GraphBasedRelationMatcher(KnowledgeRelationMatcher):
    """Relation matcher based on knowledge graphs"""

    def match(
        self,
        heads: List[KnowledgeHead],
        relations: List[KnowledgeRelation] = None,
        **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        sample_graph = kwargs.get("sample_graph")
        head_relations = []

        if sample_graph:
            matched_rels = set()

            for kg in sample_graph:
                matched_rels.add(kg.relation)

            if relations:
                matched_rels = matched_rels.intersection(set(relations))

            for head in heads:
                for relation in matched_rels:
                    head_relations.append((head, relation))

        return head_relations
