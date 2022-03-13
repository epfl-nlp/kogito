from abc import ABC, abstractmethod
from typing import List, Optional

from spacy.tokens import Doc
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

from kogito.core.head import KnowledgeHead, KnowledgeHeadType


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
                heads.append(
                    KnowledgeHead(
                        text=sentence.text,
                        type=KnowledgeHeadType.SENTENCE,
                        entity=sentence,
                    )
                )

        return heads


class NounPhraseHeadExtractor(KnowledgeHeadExtractor):
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)

        heads = []
        clean_text = []
        text_map = set()

        for token in doc:
            if token.text not in STOP_WORDS and token.pos_ != "PROPN":
                clean_text.append(token.text)

                if token.pos_ == "NOUN" and token.text not in text_map:
                    heads.append(
                        KnowledgeHead(
                            text=token.text,
                            type=KnowledgeHeadType.NOUN_PHRASE,
                            entity=token,
                        )
                    )
                    text_map.add(token.text)

        doc = self.lang(" ".join(clean_text))

        for phrase in doc.noun_chunks:
            if phrase.text not in text_map:
                heads.append(
                    KnowledgeHead(
                        text=phrase.text,
                        type=KnowledgeHeadType.NOUN_PHRASE,
                        entity=phrase,
                    )
                )
                text_map.add(phrase.text)

        return heads
