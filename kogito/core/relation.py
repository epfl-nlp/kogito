from typing import Callable, Optional
from enum import Enum
from kogito.core.head import KnowledgeHeadType
from kogito.core.utils import vp_present_participle, article, posessive


class KnowledgeRelationType(Enum):
    """
    Represents a Knowledge relation type.
    """

    TRANSOMCS = "transomcs"
    ATOMIC = "atomic"
    CONCEPTNET = "conceptnet"

    def __repr__(self):
        return str(self.value)


class KnowledgeRelation:
    """
    Represents a concept of Knowledge Relation.
    """

    def __init__(
        self,
        text: str,
        type: KnowledgeRelationType = KnowledgeRelationType.ATOMIC,
        verbalizer: Optional[Callable] = None,
        prompt: Optional[str] = None,
    ) -> None:
        """Initialize a KnowledgeRelation

        Args:
            text (str): Relation text.
            type (KnowledgeRelationType, optional): Relation type. Defaults to KnowledgeRelationType.ATOMIC.
            verbalizer (Optional[Callable], optional): Function to convert relation to natural text. Defaults to None.
            prompt (Optional[str], optional): Prompt text to use. Defaults to None.
        """
        self.text = text
        self.type = type
        self.verbalizer = verbalizer
        self.prompt = prompt

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, KnowledgeRelation)
            and self.text == other.text
            and self.type == other.type
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.text, self.type))

    def verbalize(
        self,
        head: str,
        tail: Optional[str] = None,
        include_tail: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """Convert knowledge relation into natural text.

        Args:
            head (str): Knowledge head to use.
            tail (Optional[str], optional): Knowledge tail to use if any. Defaults to None.
            include_tail (bool, optional): Whether to include tail. Defaults to False.

        Returns:
            Optional[str]: Verbalized relation.
        """
        if self.verbalizer:
            kwargs["tail"] = tail
            text = self.verbalizer(head, **kwargs).strip()
            if include_tail and tail:
                return f"{text} {tail}"
            return f"{text} "

    @classmethod
    def from_text(
        cls, text: str, type: KnowledgeRelationType = KnowledgeRelationType.ATOMIC
    ) -> "KnowledgeRelation":
        """Initialize relation from text.

        Args:
            text (str): Relation text.
            type (KnowledgeRelationType, optional): Type of relation to use. Defaults to KnowledgeRelationType.ATOMIC.

        Returns:
            KnowledgeRelation: An instance of KnowledgeRelation
        """
        for relation in KG_RELATIONS:
            if relation.text == text:
                return relation

        return cls(text, type=type)

    def __repr__(self) -> str:
        return str(self.text)

    def copy(self) -> "KnowledgeRelation":
        """Copy itself

        Returns:
            KnowledgeRelation: Copied knowledge relation
        """
        return KnowledgeRelation(text=self.text, type=self.type, verbalizer=self.verbalizer, prompt=self.prompt)


# Verbalizers
def at_location_verbalizer(head: str, **kwargs):
    text = f"You are likely to find {article(head)} {head} in"
    tail = kwargs.get("tail")
    if tail is not None:
        return f"{text} {article(tail)}"
    return text


def capable_of_verbalizer(head: str, **kwargs):
    return f"{head} can "


def causes_verbalizer(head: str, **kwargs):
    return f"Sometimes {head} causes "


def causes_desire_verbalizer(head: str, **kwargs):
    return f"{head} would make you want to "


def created_by_verbalizer(head: str, **kwargs):
    return f"{head} is created by"


def defined_as_verbalizer(head: str, **kwargs):
    return f"{head} is defined as "


def desires_verbalizer(head: str, **kwargs):
    return f"{article(head)} {head} desires"


def has_a_verbalizer(head: str, **kwargs):
    return f"{head} {posessive(head)} "


def sub_event_verbalizer(head: str, **kwargs):
    return f"While {vp_present_participle(head)}, you would "


def has_prerequisite_verbalizer(head: str, **kwargs):
    return f"{vp_present_participle(head)} requires "


def has_property_verbalizer(head: str, **kwargs):
    return f"{head} is "


def hindered_by_verbalizer(head: str, **kwargs):
    return f"{head}. This would not happen if"


def inherits_from_verbalizer(head: str, **kwargs):
    return f"{head} inherits from"


def instance_of_verbalizer(head: str, **kwargs):
    return f"{head} is an instance of"


def is_a_verbalizer(head: str, **kwargs):
    text = f"{head} is"
    tail = kwargs.get("tail")
    if tail is not None:
        return f"{text} {article(tail)}"
    return text


def made_of_verbalizer(head: str, **kwargs):
    return f"{head} is made of"


def made_up_of_verbalizer(head: str, **kwargs):
    return f"{article(head)} {head} contains"


def motivated_by_verbalizer(head: str, **kwargs):
    return f"You would {head} because you are "


def not_capable_of_verbalizer(head: str, **kwargs):
    return f"{head} is not capable of"


def not_desires_verbalizer(head: str, **kwargs):
    return f"{article(head)} {head} does not desire"


def not_has_a_verbalizer(head: str, **kwargs):
    return f"{head} does not have a"


def not_has_property_verbalizer(head: str, **kwargs):
    return f"{head} is not"


def not_is_a_verbalizer(head: str, **kwargs):
    return f"{head} is not"


def not_made_of_verbalizer(head: str, **kwargs):
    return f"{head} is not made of"


def object_use_verbalizer(head: str, **kwargs):
    return f"{article(head)} {head} can be used for"


def part_of_verbalizer(head: str, **kwargs):
    text = f"{article(head).upper()} {head} is a part of"
    tail = kwargs.get("tail")
    if tail is not None:
        return f"{text} {article(tail)}"
    return text


def receives_action_verbalizer(head: str, **kwargs):
    return f"{head} can be "


def related_to_verbalizer(head: str, **kwargs):
    return f"{head} is related to "


def symbol_of_verbalizer(head: str, **kwargs):
    return f"{head} is a symbol of"


def used_for_verbalizer(head: str, **kwargs):
    return f"{article(head).upper()} {head} is for "


def is_after_verbalizer(head: str, **kwargs):
    return f"{head}. Before that, "


def is_before_verbalizer(head: str, **kwargs):
    return f"{head}. After that, "


def is_filled_by_verbalizer(head: str, **kwargs):
    return f"{head} is filled by"


def o_effect_verbalizer(head: str, **kwargs):
    return f"{head}. The effect on others will be"


def o_react_verbalizer(head: str, **kwargs):
    return f"{head}. As a result, others feel"


def o_want_verbalizer(head: str, **kwargs):
    return f"{head}. After, others will want to"


def x_attr_verbalizer(head: str, **kwargs):
    return f"{head}. PersonX is"


def x_effect_verbalizer(head: str, **kwargs):
    return f"{head}. The effect on PersonX will be"


def x_intent_verbalizer(head: str, **kwargs):
    return f"{head}. PersonX did this to"


def x_need_verbalizer(head: str, **kwargs):
    return f"{head}. Before, PersonX needs to"


def x_react_verbalizer(head: str, **kwargs):
    return f"{head}. PersonX will be"


def x_reason_verbalizer(head: str, **kwargs):
    return f"{head}. PersonX did this because"


def x_want_verbalizer(head: str, **kwargs):
    return f"{head}. After, PersonX will want to"


# Relations
AT_LOCATION = KnowledgeRelation("AtLocation", verbalizer=at_location_verbalizer)
CAPABLE_OF = KnowledgeRelation("CapableOf", verbalizer=capable_of_verbalizer)
CAUSES = KnowledgeRelation("Causes", verbalizer=causes_verbalizer)
CAUSES_DESIRE = KnowledgeRelation(
    "CausesDesire",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=causes_desire_verbalizer,
)
CREATED_BY = KnowledgeRelation(
    "CreatedBy", type=KnowledgeRelationType.CONCEPTNET, verbalizer=created_by_verbalizer
)
DEFINED_AS = KnowledgeRelation(
    "DefinedAs", type=KnowledgeRelationType.CONCEPTNET, verbalizer=defined_as_verbalizer
)
DESIRE_OF = KnowledgeRelation("DesireOf", type=KnowledgeRelationType.CONCEPTNET)
DESIRES = KnowledgeRelation("Desires", verbalizer=desires_verbalizer)
HAS_A = KnowledgeRelation(
    "HasA", type=KnowledgeRelationType.CONCEPTNET, verbalizer=has_a_verbalizer
)
HAS_FIRST_SUB_EVENT = KnowledgeRelation(
    "HasFirstSubevent",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=sub_event_verbalizer,
)
HAS_LAST_SUB_EVENT = KnowledgeRelation(
    "HasLastSubevent",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=sub_event_verbalizer,
)
HAS_PAIN_CHARACTER = KnowledgeRelation(
    "HasPainCharacter", type=KnowledgeRelationType.CONCEPTNET
)
HAS_PAIN_INTENSITY = KnowledgeRelation(
    "HasPainIntensity", type=KnowledgeRelationType.CONCEPTNET
)
HAS_PREREQUISITE = KnowledgeRelation(
    "HasPrerequisite",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=has_prerequisite_verbalizer,
)
HAS_PROPERTY = KnowledgeRelation("HasProperty", verbalizer=has_property_verbalizer)
HAS_SUB_EVENT = KnowledgeRelation("HasSubEvent", verbalizer=sub_event_verbalizer)
HAS_SUB_EVENT2 = KnowledgeRelation(
    "HasSubevent",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=sub_event_verbalizer,
)
HINDERED_BY = KnowledgeRelation("HinderedBy", verbalizer=hindered_by_verbalizer)
INHERITS_FROM = KnowledgeRelation(
    "InheritsFrom",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=inherits_from_verbalizer,
)
INSTANCE_OF = KnowledgeRelation(
    "InstanceOf",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=instance_of_verbalizer,
)
IS_A = KnowledgeRelation(
    "IsA", type=KnowledgeRelationType.CONCEPTNET, verbalizer=is_a_verbalizer
)
LOCATED_NEAR = KnowledgeRelation("LocatedNear", type=KnowledgeRelationType.CONCEPTNET)
LOCATION_OF_ACTION = KnowledgeRelation(
    "LocationOfAction", type=KnowledgeRelationType.CONCEPTNET
)
MADE_OF = KnowledgeRelation(
    "MadeOf", type=KnowledgeRelationType.CONCEPTNET, verbalizer=made_of_verbalizer
)
MADE_UP_OF = KnowledgeRelation("MadeUpOf", verbalizer=made_up_of_verbalizer)
MOTIVATED_BY_GOAL = KnowledgeRelation(
    "MotivatedByGoal",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=motivated_by_verbalizer,
)
NOT_CAPABLE_OF = KnowledgeRelation(
    "NotCapableOf",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=not_capable_of_verbalizer,
)
NOT_DESIRES = KnowledgeRelation("NotDesires", verbalizer=not_desires_verbalizer)
NOT_HAS_A = KnowledgeRelation(
    "NotHasA", type=KnowledgeRelationType.CONCEPTNET, verbalizer=not_has_a_verbalizer
)
NOT_HAS_PROPERTY = KnowledgeRelation(
    "NotHasProperty",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=not_has_property_verbalizer,
)
NOT_IS_A = KnowledgeRelation(
    "NotIsA", type=KnowledgeRelationType.CONCEPTNET, verbalizer=not_is_a_verbalizer
)
NOT_MADE_OF = KnowledgeRelation(
    "NotMadeOf",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=not_made_of_verbalizer,
)
OBJECT_USE = KnowledgeRelation("ObjectUse", verbalizer=object_use_verbalizer)
PART_OF = KnowledgeRelation(
    "PartOf", type=KnowledgeRelationType.CONCEPTNET, verbalizer=part_of_verbalizer
)
RECEIVES_ACTION = KnowledgeRelation(
    "ReceivesAction",
    type=KnowledgeRelationType.CONCEPTNET,
    verbalizer=receives_action_verbalizer,
)
RELATED_TO = KnowledgeRelation(
    "RelatedTo", type=KnowledgeRelationType.CONCEPTNET, verbalizer=related_to_verbalizer
)
SYMBOL_OF = KnowledgeRelation(
    "SymbolOf", type=KnowledgeRelationType.CONCEPTNET, verbalizer=symbol_of_verbalizer
)
USED_FOR = KnowledgeRelation(
    "UsedFor", type=KnowledgeRelationType.CONCEPTNET, verbalizer=used_for_verbalizer
)
IS_AFTER = KnowledgeRelation("isAfter", verbalizer=is_after_verbalizer)
IS_BEFORE = KnowledgeRelation("isBefore", verbalizer=is_before_verbalizer)
IS_FILLED_BY = KnowledgeRelation("isFilledBy", verbalizer=is_filled_by_verbalizer)
O_EFFECT = KnowledgeRelation("oEffect", verbalizer=o_effect_verbalizer)
O_REACT = KnowledgeRelation("oReact", verbalizer=o_react_verbalizer)
O_WANT = KnowledgeRelation("oWant", verbalizer=o_want_verbalizer)
X_ATTR = KnowledgeRelation("xAttr", verbalizer=x_attr_verbalizer)
X_EFFECT = KnowledgeRelation("xEffect", verbalizer=x_effect_verbalizer)
X_INTENT = KnowledgeRelation("xIntent", verbalizer=x_intent_verbalizer)
X_NEED = KnowledgeRelation(
    "xNeed",
    verbalizer=x_need_verbalizer,
    prompt="What needs to be true for this event to take place?",
)
X_REACT = KnowledgeRelation("xReact", verbalizer=x_react_verbalizer)
X_REASON = KnowledgeRelation("xReason", verbalizer=x_reason_verbalizer)
X_WANT = KnowledgeRelation("xWant", verbalizer=x_want_verbalizer)

KG_RELATIONS = [
    AT_LOCATION,
    CAPABLE_OF,
    CAUSES,
    CAUSES_DESIRE,
    CREATED_BY,
    DEFINED_AS,
    DESIRE_OF,
    DESIRES,
    HAS_A,
    HAS_FIRST_SUB_EVENT,
    HAS_LAST_SUB_EVENT,
    HAS_PAIN_CHARACTER,
    HAS_PAIN_INTENSITY,
    HAS_PREREQUISITE,
    HAS_PROPERTY,
    HAS_SUB_EVENT,
    HAS_SUB_EVENT2,
    HINDERED_BY,
    INHERITS_FROM,
    INSTANCE_OF,
    IS_A,
    LOCATED_NEAR,
    LOCATION_OF_ACTION,
    MADE_OF,
    MADE_UP_OF,
    MOTIVATED_BY_GOAL,
    NOT_CAPABLE_OF,
    NOT_DESIRES,
    NOT_HAS_A,
    NOT_HAS_PROPERTY,
    NOT_IS_A,
    NOT_MADE_OF,
    OBJECT_USE,
    PART_OF,
    RECEIVES_ACTION,
    RELATED_TO,
    SYMBOL_OF,
    USED_FOR,
    IS_AFTER,
    IS_BEFORE,
    IS_FILLED_BY,
    O_EFFECT,
    O_REACT,
    O_WANT,
    X_ATTR,
    X_EFFECT,
    X_INTENT,
    X_NEED,
    X_REACT,
    X_REASON,
    X_WANT,
]

CONCEPTNET_RELATIONS = [
    relation
    for relation in KG_RELATIONS
    if relation.type == KnowledgeRelationType.CONCEPTNET
]

ATOMIC_RELATIONS = [
    relation
    for relation in KG_RELATIONS
    if relation.type == KnowledgeRelationType.ATOMIC
]

#: ATOMIC 2020 Physical relations
PHYSICAL_RELATIONS = [
    OBJECT_USE,
    CAPABLE_OF,
    MADE_UP_OF,
    HAS_PROPERTY,
    DESIRES,
    NOT_DESIRES,
    AT_LOCATION,
]

#: ATOMIC 2020 Event relations
EVENT_RELATIONS = [
    CAUSES,
    HINDERED_BY,
    X_REASON,
    IS_AFTER,
    IS_BEFORE,
    HAS_SUB_EVENT,
    IS_FILLED_BY,
]

#: ATOMIC 2020 Social relations
SOCIAL_RELATIONS = [
    X_INTENT,
    X_REACT,
    O_REACT,
    X_ATTR,
    X_EFFECT,
    X_NEED,
    X_WANT,
    O_EFFECT,
    O_WANT,
]

NOUN_PHRASE_RELATIONS = PHYSICAL_RELATIONS
SENTENCE_RELATIONS = EVENT_RELATIONS + SOCIAL_RELATIONS
VERB_PHRASE_RELATIONS = EVENT_RELATIONS

HEAD_TO_RELATION_MAP = {
    KnowledgeHeadType.TEXT: SENTENCE_RELATIONS,
    KnowledgeHeadType.SENTENCE: SENTENCE_RELATIONS,
    KnowledgeHeadType.NOUN_PHRASE: NOUN_PHRASE_RELATIONS,
    KnowledgeHeadType.VERB_PHRASE: VERB_PHRASE_RELATIONS,
}

CONCEPTNET_TO_ATOMIC_MAP = {
    CAUSES: [CAUSES, X_EFFECT],
    CAUSES_DESIRE: X_WANT,
    MADE_OF: MADE_UP_OF,
    HAS_A: [MADE_UP_OF, HAS_PROPERTY],
    HAS_PREREQUISITE: X_NEED,
    HAS_SUB_EVENT2: HAS_SUB_EVENT,
    HAS_FIRST_SUB_EVENT: HAS_SUB_EVENT,
    HAS_LAST_SUB_EVENT: HAS_SUB_EVENT,
    MOTIVATED_BY_GOAL: [X_INTENT, X_REASON],
    PART_OF: MADE_UP_OF,
    USED_FOR: OBJECT_USE,
    RECEIVES_ACTION: [MADE_UP_OF, AT_LOCATION, CAUSES, OBJECT_USE],
}


def register_relation(relation: KnowledgeRelation):
    if relation not in KG_RELATIONS:
        KG_RELATIONS.append(relation)
