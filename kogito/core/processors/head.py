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

        for token in doc:
            if token.text not in STOP_WORDS and token.pos_ != "PROPN":
                clean_text.append(token.text)

                if token.pos_ == "NOUN":
                    heads.append(
                        KnowledgeHead(
                            text=token.text,
                            type=KnowledgeHeadType.NOUN_PHRASE,
                            entity=token,
                        )
                    )

        doc = self.lang(" ".join(clean_text))

        for phrase in doc.noun_chunks:
            heads.append(
                KnowledgeHead(
                    text=phrase.text,
                    type=KnowledgeHeadType.NOUN_PHRASE,
                    entity=phrase,
                )
            )

        return heads


class VerbPhraseHeadExtractor(KnowledgeHeadExtractor):
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)

        heads = []

        for token in doc:
            if token.dep_ == 'ROOT':
                heads.append(
                    KnowledgeHead(
                        text=f"to {token.lemma_}",
                        type=KnowledgeHeadType.VERB_PHRASE,
                        entity=token
                    ))
                
                for child in token.children:
                    if child.dep_ == 'dobj':
                        heads.append(
                            KnowledgeHead(
                                text=f"{token.lemma_} {child.text}",
                                type=KnowledgeHeadType.VERB_PHRASE,
                                entity=[token, child],
                            )
                        )

        return heads