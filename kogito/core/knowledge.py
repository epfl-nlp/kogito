
from typing import List
from enum import Enum

from kogito.core.utils import vp_present_participle, article, posessive

ATOMIC_RELATIONS = [
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

EOS_TOKEN = '[EOS]'
GEN_TOKEN = '[GEN]'
PAD_TOKEN = '[PAD]'

DECODE_METHODS = ['greedy', 'beam']


class KnowledgeBase(Enum):
    TRANSOMCS  = 'transomcs'
    ATOMIC     = 'atomic'
    CONCEPTNET = 'conceptnet'
    ATOMIC2020 = 'atomic2020'


class UnknownRelationError(Exception):
    pass


class Knowledge:
    def __init__(self, head: str, relation: str, tails: List[str] = None, base: KnowledgeBase = KnowledgeBase.ATOMIC2020):
        self.head = head
        self.relation = relation
        self.tails = tails
        self.base = base
        self.prompt = None
    
    def to_prompt(self):
        head = self.head
        relation = self.relation
        tail = self.tails[0] if self.tails else ''

        if self.base == KnowledgeBase.CONCEPTNET or self.base == KnowledgeBase.TRANSOMCS:
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
            elif relation == "HasFirstSubevent" or relation == "HasSubevent" or relation == "HasLastSubevent":
                prompt = "While {}, you would ".format(vp_present_participle(head))
            elif relation == "InheritsFrom":
                prompt = "{} inherits from".format(head)
            elif relation == "PartOf":
                prompt = "{} {} is a part of {} ".format(article(head).upper(), head, article(tail))
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
                prompt = "{} is filled by".format(head) #TODO
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

    def to_query(self, decode_method: str = 'greedy'):
        if decode_method == 'greedy':
            return "{} {}".format(self.head, self.relation)
        elif decode_method == 'beam':
            return "{} {} [GEN]".format(self.head, self.relation)
        else:
            raise ValueError