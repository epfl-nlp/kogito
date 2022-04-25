from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np
import torch
import pandas as pd
from torch import nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from spacy.language import Language
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader

from kogito.core.head import KnowledgeHead
from kogito.core.relation import (
    KnowledgeRelation,
    HEAD_TO_RELATION_MAP,
    PHYSICAL_RELATIONS,
    EVENT_RELATIONS,
    SOCIAL_RELATIONS,
)

from kogito.core.processors.models.swem import SWEMHeadDataset, SWEMClassifier

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


class SWEMRelationMatcher(KnowledgeRelationMatcher):
    def match(
        self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        vocab = np.load("./data/vocab_glove_100d.npy", allow_pickle=True).item()
        df = pd.DataFrame({"text": [str(head) for head in heads]})
        dataset = SWEMHeadDataset(df, vocab, lang=self.lang)
        model = None
        model.load_state_dict(
            torch.load("./models/swem_multi_label_finetune_state_dict.pth")
        )
        probs = model.forward(head_inputs).detach().numpy()
        head_relations = []

        for head, prob in zip(heads, probs):
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


class DistilBertHeadDataset(Dataset):
    def __init__(self, heads):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.texts = [
            self.tokenizer(
                head.text,
                padding="max_length",
                max_length=32,
                truncation=True,
                return_tensors="pt",
            )
            for head in heads
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class DistilBERTClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.classifier = nn.Sequential(self.dropout, self.linear)

    def forward(self, input_ids, mask):
        outputs = self.distilbert(
            input_ids=input_ids, attention_mask=mask, return_dict=False
        )
        outputs = self.classifier(outputs[0][:, 0, :])
        return outputs

    def predict_step(self, batch, batch_idx):
        X = batch
        mask = X["attention_mask"]
        input_ids = X["input_ids"].squeeze(1)
        outputs = self.forward(input_ids, mask)
        probs = torch.sigmoid(outputs)
        return probs


class DistilBertRelationMatcher(KnowledgeRelationMatcher):
    def match(
        self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs
    ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
        dataset = DistilBertHeadDataset(heads)
        dataloader = DataLoader(dataset, batch_size=128)
        model = DistilBERTClassifier.load_from_checkpoint(
            "./models/distilbert/distilbert_model_20220404H1852.ckpt"
        )
        trainer = pl.Trainer()
        probs = [
            prob
            for batch_probs in trainer.predict(model, dataloader)
            for prob in batch_probs
        ]
        head_relations = []

        for head, prob in zip(heads, probs):
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
