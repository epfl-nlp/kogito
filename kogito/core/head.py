from typing import Any
from enum import Enum


class KnowledgeHeadType(Enum):
    SENTENCE = "sentence"
    NOUN_PHRASE = "noun_phrase"


class KnowledgeHead:
    def __init__(self, text: str, type: KnowledgeHeadType, entity: Any = None) -> None:
        self.text = text
        self.type = type
        self.entity = entity
