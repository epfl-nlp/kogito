from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import partial

import numpy as np
import torch
import pytorch_lightning as pl
from spacy.language import Language
from torch.utils.data import DataLoader

from kogito.core.head import KnowledgeHead
from kogito.core.relation import (
    KnowledgeRelation,
    HEAD_TO_RELATION_MAP,
    PHYSICAL_RELATIONS,
    EVENT_RELATIONS,
    SOCIAL_RELATIONS,
)

from kogito.core.processors.models.swem import SWEMHeadDataset, SWEMClassifier
from kogito.core.processors.models.distilbert import DistilBERTHeadDataset, DistilBERTClassifier
from kogito.core.processors.models.bert import BERTHeadDataset, BERTClassifier

RELATION_CLASSES = [PHYSICAL_RELATIONS, EVENT_RELATIONS, SOCIAL_RELATIONS]


class KnowledgeRelationMatcher(ABC):
    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        self.name = name
        self.lang = lang

    @abstractmethod
    def match(
        self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        raise NotImplementedError


class SimpleRelationMatcher(KnowledgeRelationMatcher):
    def match(
        self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs
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
    def __init__(self, name: str, dataset_class, model_class, model_path: str, batch_size=64, lang: Optional[Language] = None) -> None:
        super().__init__(name, lang)
        self.dataset_class = dataset_class
        self.model_class = model_class
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = model_class.from_pretrained(model_path)

    def match(self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        data = [str(head) for head in heads]
        dataset = self.dataset_class(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        trainer = pl.Trainer()
        predictions = torch.cat(trainer.predict(self.model, dataloaders=dataloader)).numpy()
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
    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        vocab = np.load("./data/vocab_glove_100d.npy", allow_pickle=True).item()
        dataset_class = partial(SWEMHeadDataset, vocab=vocab, lang=lang)
        model_class = SWEMClassifier
        model_path = "mismayil/kogito-rc-swem"
        super().__init__(name, dataset_class=dataset_class, model_class=model_class, model_path=model_path, lang=lang)



class DistilBERTRelationMatcher(ModelBasedRelationMatcher):
    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        dataset_class = DistilBERTHeadDataset
        model_class = DistilBERTClassifier
        model_path = "mismayil/kogito-rc-distilbert"
        super().__init__(name, dataset_class=dataset_class, model_class=model_class, model_path=model_path, lang=lang)


class BERTRelationMatcher(ModelBasedRelationMatcher):
    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        dataset_class = BERTHeadDataset
        model_class = BERTClassifier
        model_path = "mismayil/kogito-rc-bert"
        super().__init__(name, dataset_class=dataset_class, model_class=model_class, model_path=model_path, lang=lang)


class GraphBasedRelationMatcher(KnowledgeRelationMatcher):
    def match(
        self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs
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
