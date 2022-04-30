from abc import ABC, abstractmethod
from typing import List, Optional

from spacy.tokens import Doc
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

from kogito.core.head import KnowledgeHead, KnowledgeHeadType
from kogito.core.utils import IGNORE_WORDS


class KnowledgeHeadExtractor(ABC):
    """Base class for head extraction"""

    def __init__(self, name: str, lang: Optional[Language] = None) -> None:
        """Initialize a head extractor

        Args:
            name (str): Unique head extractor name
            lang (Optional[Language], optional): Spacy language pipeline to use. Defaults to None.
        """
        self.name = name
        self.lang = lang

    @abstractmethod
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        """Extract heads from text

        Args:
            text (str): Text to extract from
            doc (Optional[Doc], optional): Spacy doc to use for extraction. Defaults to None.

        Raises:
            NotImplementedError: This method has to be implemented by
                                 concrete subclasses.

        Returns:
            List[KnowledgeHead]: List of extracted knowledge heads.
        """
        raise NotImplementedError


class SentenceHeadExtractor(KnowledgeHeadExtractor):
    """Extracts sentences as heads from text"""

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
    """Extracts noun phrases as heads from text"""

    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)

        heads = []
        clean_text = []

        for token in doc:
            if (
                token.text not in STOP_WORDS.union(IGNORE_WORDS)
                and token.pos_ != "PROPN"
            ):
                clean_text.append(token.text)

                if token.pos_ == "NOUN":
                    heads.append(
                        KnowledgeHead(
                            text=token.text,
                            type=KnowledgeHeadType.NOUN_PHRASE,
                            entity=token,
                        )
                    )

        for phrase in doc.noun_chunks:
            clean_phrase = []
            phrase_doc = self.lang(phrase.text)

            for token in phrase_doc:
                if token.text not in STOP_WORDS.union(IGNORE_WORDS):
                    clean_phrase.append(token.text)

            clean_text = " ".join(clean_phrase).strip()

            if clean_text:
                heads.append(
                    KnowledgeHead(
                        text=clean_text,
                        type=KnowledgeHeadType.NOUN_PHRASE,
                        entity=phrase,
                    )
                )

        return heads


class VerbPhraseHeadExtractor(KnowledgeHeadExtractor):
    """Extracts verb phrases as heads from text"""

    def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
        if not doc:
            doc = self.lang(text)

        heads = []

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                heads.append(
                    KnowledgeHead(
                        text=f"to {token.lemma_}",
                        type=KnowledgeHeadType.VERB_PHRASE,
                        entity=token,
                    )
                )

                for child in token.children:
                    if child.dep_ in ("attr", "dobj"):
                        heads.append(
                            KnowledgeHead(
                                text=f"{token.lemma_} {child.text}",
                                type=KnowledgeHeadType.VERB_PHRASE,
                                entity=[token, child],
                            )
                        )

        return heads
