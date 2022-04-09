from typing import Any, Callable
from enum import Enum


class KnowledgeHeadType(Enum):
    TEXT = "text"
    SENTENCE = "sentence"
    NOUN_PHRASE = "noun_phrase"
    VERB_PHRASE = "verb_phrase"


class KnowledgeHead:
    def __init__(self, text: str, type: KnowledgeHeadType = KnowledgeHeadType.TEXT, entity: Any = None, verbalizer: Callable = None) -> None:
        self.text = text
        self.type = type
        self.entity = entity
        self.verbalizer = verbalizer

    def __eq__(self, other: object) -> bool:
        if self == other:
            return True
        return isinstance(other, KnowledgeHead) and self.text == other.text

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.text)

    def verbalize(self):
        if self.verbalizer:
            return self.verbalizer(self.text)

    def __repr__(self):
        return str(self.text)


def head_verbalizer(head: str):
    return head