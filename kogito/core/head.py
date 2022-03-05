from abc import ABC, abstractmethod
from typing import List, Optional, Any
from enum import Enum

import pytextrank

from spacy.tokens import Doc
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

class KnowledgeHeadType(Enum):
    SENTENCE = "sentence"
    PHRASE = "phrase"
    NOUN = "noun"


class KnowledgeHead:
    def __init__(self, text: str, type: KnowledgeHeadType, entity: Any = None) -> None:
        self.text = text
        self.type = type
        self.entity = entity


class KnowledgeHeadExtractor(ABC):
    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        self.name = name
        self.lang = lang

    @abstractmethod
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        raise NotImplementedError


class SentenceHeadExtractor(KnowledgeHeadExtractor):
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)
        
        heads = []

        for sentence in doc.sents:
            if sentence.text.strip():
                heads.append(KnowledgeHead(text=sentence.text, type=KnowledgeHeadType.SENTENCE, entity=sentence))
        
        return heads


class PhraseHeadExtractor(KnowledgeHeadExtractor):
    def __init__(self, name: str, lang: Optional[Language] = None, min_phrase_rank: float = 0.01) -> None:
        super().__init__(name, lang)
        self.min_phrase_rank = min_phrase_rank

    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)
        
        textrank = self.lang.add_pipe("textrank")
        doc = textrank(doc)

        heads = []
        tokenizer = self.lang.tokenizer

        for phrase in doc._.phrases:
            if phrase.rank >= self.min_phrase_rank and phrase.text.strip():
                tokens = tokenizer(phrase.text)
                phrase.text = " ".join([token.text for token in tokens if token.text not in STOP_WORDS])
                heads.append(KnowledgeHead(text=phrase.text, type=KnowledgeHeadType.PHRASE, entity=phrase))
        
        return heads
    

class NounHeadExtractor(KnowledgeHeadExtractor):
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)

        heads = []

        for token in doc:
            if token.pos_ == "NOUN":
                heads.append(KnowledgeHead(text=token.text, type=KnowledgeHeadType.NOUN, entity=token))
        
        return heads