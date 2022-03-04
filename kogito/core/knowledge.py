from typing import List
from enum import Enum
import pandas as pd
import json

from kogito.core.utils import vp_present_participle, article, posessive

KG_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]

CONCEPTNET_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPrerequisite",
    "HasProperty",
    "HasSubevent",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "MadeOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "PartOf",
    "ReceivesAction",
    "SymbolOf",
    "UsedFor"
]

ATOMIC_RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "Desires",
    "HasProperty",
    "HasSubEvent",
    "HinderedBy",
    "MadeUpOf",
    "NotDesires",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]

EOS_TOKEN = "[EOS]"
GEN_TOKEN = "[GEN]"
PAD_TOKEN = "[PAD]"

DECODE_METHODS = ["greedy", "beam"]


class KnowledgeBase(Enum):
    TRANSOMCS = "transomcs"
    ATOMIC = "atomic"
    CONCEPTNET = "conceptnet"
    ATOMIC2020 = "atomic2020"

    def __repr__(self):
        return str(self.value)


class UnknownRelationError(Exception):
    pass


class Knowledge:
    def __init__(
        self,
        head: str = None,
        relation: str = None,
        tails: List[str] = None,
        base: KnowledgeBase = KnowledgeBase.ATOMIC2020,
    ):
        self.head = head
        self.relation = relation
        self.tails = tails or []
        self.base = base
        self.prompt = None

    def __repr__(self):
        return f'Knowledge(head="{self.head}", relation="{self.relation}", tails={self.tails}, base={self.base})'

    def to_prompt(self):
        head = self.head
        relation = self.relation
        tail = self.tails[0] if self.tails else ""

        if (
            self.base == KnowledgeBase.CONCEPTNET
            or self.base == KnowledgeBase.TRANSOMCS
        ):
            if relation == "AtLocation":
                prompt = "You are likely to find {} {} in {} ".format(
                    article(head), head, article(tail)
                )
            elif relation == "CapableOf":
                prompt = "{} can ".format(head)
            elif relation == "CausesDesire":
                prompt = "{} would make you want to ".format(head)
            elif relation == "Causes":
                prompt = "Sometimes {} causes ".format(head)
            elif relation == "CreatedBy":
                prompt = "{} is created by".format(head)
            elif relation == "Desires":
                prompt = "{} {} desires".format(article(head), head)
            elif relation == "HasA":
                prompt = "{} {} ".format(head, posessive(head))
            elif relation == "HasPrerequisite":
                prompt = "{} requires ".format(vp_present_participle(head))
            elif relation == "HasProperty":
                prompt = "{} is ".format(head)
            elif relation == "MotivatedByGoal":
                prompt = "You would {} because you are ".format(head)
            elif relation == "ReceivesAction":
                prompt = "{} can be ".format(head)
            elif relation == "UsedFor":
                prompt = "{} {} is for ".format(article(head).upper(), head)
            elif (
                relation == "HasFirstSubevent"
                or relation == "HasSubevent"
                or relation == "HasLastSubevent"
            ):
                prompt = "While {}, you would ".format(vp_present_participle(head))
            elif relation == "InheritsFrom":
                prompt = "{} inherits from".format(head)
            elif relation == "PartOf":
                prompt = "{} {} is a part of {} ".format(
                    article(head).upper(), head, article(tail)
                )
            elif relation == "IsA":
                prompt = "{} is {} ".format(head, article(tail))
            elif relation == "InstanceOf":
                prompt = "{} is an instance of".format(head)
            elif relation == "MadeOf":
                prompt = "{} is made of".format(head)
            elif relation == "DefinedAs":
                prompt = "{} is defined as ".format(head)
            elif relation == "NotCapableOf":
                prompt = "{} is not capable of".format(head)
            elif relation == "NotDesires":
                prompt = "{} {} does not desire".format(article(head), head)
            elif relation == "NotHasA":
                prompt = "{} does not have a".format(head)
            elif relation == "NotHasProperty" or relation == "NotIsA":
                prompt = "{} is not".format(head)
            elif relation == "NotMadeOf":
                prompt = "{} is not made of".format(head)
            elif relation == "SymbolOf":
                prompt = "{} is a symbol of".format(head)
            else:
                raise UnknownRelationError(relation)
        elif self.base == KnowledgeBase.ATOMIC or self.base == KnowledgeBase.ATOMIC2020:
            if relation == "AtLocation":
                prompt = "You are likely to find {} {} in {} ".format(
                    article(head), head, article(tail)
                )
            elif relation == "CapableOf":
                prompt = "{} can ".format(head)
            elif relation == "Causes":
                prompt = "Sometimes {} causes ".format(head)
            elif relation == "Desires":
                prompt = "{} {} desires".format(article(head), head)
            elif relation == "HasProperty":
                prompt = "{} is ".format(head)
            elif relation == "HasSubEvent":
                prompt = "While {}, you would ".format(vp_present_participle(head))
            elif relation == "HinderedBy":
                prompt = "{}. This would not happen if"
            elif relation == "MadeUpOf":
                prompt = "{} {} contains".format(article(head), head)
            elif relation == "NotDesires":
                prompt = "{} {} does not desire".format(article(head), head)
            elif relation == "ObjectUse":
                prompt = "{} {} can be used for".format(article(head), head)
            elif relation == "isAfter":
                prompt = "{}. Before that, ".format(head)
            elif relation == "isBefore":
                prompt = "{}. After that, ".format(head)
            elif relation == "isFilledBy":
                prompt = "{} is filled by".format(head)  # TODO
            elif relation == "oEffect":
                prompt = "{}. The effect on others will be".format(head)
            elif relation == "oReact":
                prompt = "{}. As a result, others feel".format(head)
            elif relation == "oWant":
                prompt = "{}. After, others will want to".format(head)
            elif relation == "xAttr":
                prompt = "{}. PersonX is".format(head)
            elif relation == "xEffect":
                prompt = "{}. The effect on PersonX will be".format(head)
            elif relation == "xIntent":
                prompt = "{}. PersonX did this to".format(head)
            elif relation == "xNeed":
                prompt = "{}. Before, PersonX needs to".format(head)
            elif relation == "xReact":
                prompt = "{}. PersonX will be".format(head)
            elif relation == "xReason":
                prompt = "{}. PersonX did this because".format(head)
            elif relation == "xWant":
                prompt = "{}. After, PersonX will want to".format(head)
        else:
            raise UnknownRelationError(relation)

        self.prompt = prompt
        return prompt.strip()

    def to_query(self, decode_method: str = "greedy"):
        if decode_method == "greedy":
            return "{} {}".format(self.head, self.relation)
        elif decode_method == "beam":
            return "{} {} [GEN]".format(self.head, self.relation)
        else:
            raise ValueError

    def copy(self):
        return Knowledge(
            base=self.base, head=self.head, relation=self.relation, tails=self.tails
        )

    def to_json(self, only_one_tail=False):
        return {
            "head": self.head,
            "relation": self.relation,
            "tails": self.tails[0] if self.tails and only_one_tail else self.tails,
        }


class KnowledgeGraph:
    def __init__(self, graph: List[Knowledge]):
        self.graph = graph
        self._graph_iter = None

    def __iter__(self):
        self._graph_iter = iter(self.graph)
        return self

    def __next__(self):
        return next(self._graph_iter)

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, idx):
        return self.graph[idx]

    @classmethod
    def from_jsonl(
        cls,
        filepath: str,
        base: KnowledgeBase = KnowledgeBase.ATOMIC2020,
        head_attr: str = "head",
        relation_attr: str = "relation",
        tails_attr: str = "tails",
    ):
        kg_list = []

        with open(filepath) as file:
            for line in file:
                kg_json = json.loads(line)
                head = kg_json.get(head_attr)
                relation = kg_json.get(relation_attr)
                tails = kg_json.get(tails_attr)
                kg_list.append(
                    Knowledge(base=base, head=head, relation=relation, tails=tails)
                )

        return cls(kg_list)

    def to_jsonl(self, filepath):
        with open(filepath, "w") as file:
            lines = []
            for kg in self.graph:
                lines.append(json.dumps(kg.to_json()))
            file.writelines("\n".join(lines))

    def to_dataframe(self):
        return pd.DataFrame([kg.to_json(only_one_tail=True) for kg in self.graph])
