from abc import ABC, abstractmethod
from typing import List, Tuple

from kogito.core.head import KnowledgeHead
from kogito.core.relation import HEAD_TO_RELATION_MAP


class KnowledgeRelationMatcher(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def match(self, heads: List[KnowledgeHead], relations: List[str] = None) -> List[Tuple[KnowledgeHead, str]]:
        raise NotImplementedError


class SimpleRelationMatcher(KnowledgeRelationMatcher):
    def match(self, heads: List[KnowledgeHead], relations: List[str] = None) -> List[Tuple[KnowledgeHead, str]]:
        head_relations = []

        for head in heads:
            rels_to_match = HEAD_TO_RELATION_MAP[head.type]
            if relations:
                rels_to_match = set(HEAD_TO_RELATION_MAP[head.type]).intersection(set(relations))
            for relation in rels_to_match:
                head_relations.append((head, relation))

        return head_relations
