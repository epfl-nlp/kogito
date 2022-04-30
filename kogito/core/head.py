from typing import Any, Callable, Optional
from enum import Enum


class KnowledgeHeadType(Enum):
    """
    Type of a Knowledge Head
    """

    TEXT = "text"
    SENTENCE = "sentence"
    NOUN_PHRASE = "noun_phrase"
    VERB_PHRASE = "verb_phrase"


class KnowledgeHead:
    """
    Represents a concept of Knowledge Head.
    """

    def __init__(
        self,
        text: str,
        type: KnowledgeHeadType = KnowledgeHeadType.TEXT,
        entity: Any = None,
        verbalizer: Optional[Callable] = None,
    ) -> None:
        """Initialize a Knowledge Head.

        Args:
            text (str): Head text.
            type (KnowledgeHeadType, optional): Type of a Knowledge head. Defaults to KnowledgeHeadType.TEXT.
            entity (Any, optional): External Knowledge head entity. Defaults to None.
            verbalizer (Optional[Callable], optional): Function to convert knowledge head to natural text.
                                                      Defaults to None.
        """
        self.text = text
        self.type = type
        self.entity = entity
        self.verbalizer = verbalizer

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KnowledgeHead) and self.text == other.text

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.text)

    def verbalize(self) -> Optional[str]:
        """Convert head to a meaningful text.

        Returns:
            Optional[str]: Verbalized head
        """
        if self.verbalizer:
            return self.verbalizer(self.text)

    def __repr__(self) -> str:
        return str(self.text)


def head_verbalizer(head: str):
    return head
