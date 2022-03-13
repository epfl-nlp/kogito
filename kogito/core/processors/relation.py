from abc import ABC, abstractmethod
from typing import List, Tuple

from kogito.core.head import KnowledgeHead
from kogito.core.relation import HEAD_TO_RELATION_MAP


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
