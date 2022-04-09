from typing import Any
from enum import Enum


class KnowledgeHeadType(Enum):
    TEXT = "text"
    SENTENCE = "sentence"
    NOUN_PHRASE = "noun_phrase"
    VERB_PHRASE = "verb_phrase"


class KnowledgeHead:
    def __init__(self, text: str, type: KnowledgeHeadType = KnowledgeHeadType.TEXT, entity: Any = None) -> None:
        self.text = text
        self.type = type
        self.entity = entity
    
    def verbalize(self):
        return f"Event: {self.text}"

    def __repr__(self):
        return str(self.text)
