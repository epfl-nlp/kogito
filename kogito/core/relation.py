from abc import ABC, abstractmethod
from typing import List, Tuple

from kogito.core.head import KnowledgeHead, KnowledgeHeadType


SENTENCE_RELATIONS = [
    "Causes",
    "CausesDesire",
    "Desires",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPrerequisite",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "ReceivesAction",
    "SymbolOf",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]

PHRASE_NOUN_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
]

HEAD_TO_RELATION_MAP = {
    KnowledgeHeadType.SENTENCE: SENTENCE_RELATIONS,
    KnowledgeHeadType.PHRASE: PHRASE_NOUN_RELATIONS,
    KnowledgeHeadType.NOUN: PHRASE_NOUN_RELATIONS,
}


class KnowledgeRelationMatcher(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def match(self, heads: List[KnowledgeHead]) -> List[Tuple[KnowledgeHead, str]]:
        raise NotImplementedError


class SimpleRelationMatcher(KnowledgeRelationMatcher):
    def match(self, heads: List[KnowledgeHead]) -> List[Tuple[KnowledgeHead, str]]:
        head_relations = []

        for head in heads:
            relations = HEAD_TO_RELATION_MAP[head.type]
            for relation in relations:
                head_relations.append((head, relation))

        return head_relations
