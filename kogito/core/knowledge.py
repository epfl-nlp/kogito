from typing import List, Union
import pandas as pd
import json

from kogito.core.relation import KnowledgeRelation, KnowledgeRelationType
from kogito.core.head import KnowledgeHead

EOS_TOKEN = "[EOS]"
GEN_TOKEN = "[GEN]"
PAD_TOKEN = "[PAD]"

DECODE_METHODS = ["greedy", "beam"]


class Knowledge:
    def __init__(
        self,
        head: Union[KnowledgeHead, str] = None,
        relation: Union[KnowledgeRelation, str] = None,
        tails: List[str] = None,
    ):
        self.head = head if isinstance(head, KnowledgeHead) else KnowledgeHead(head)
        self.relation = (
            relation
            if isinstance(relation, KnowledgeRelation)
            else KnowledgeRelation.from_text(relation)
        )
        self.tails = tails or []
        if isinstance(self.tails, str):
            self.tails = [self.tails]

    def __repr__(self):
        return f'Knowledge(head="{str(self.head)}", relation="{str(self.relation)}", tails={self.tails})'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Knowledge)
            and self.head == other.head
            and self.relation == other.relation
            and self.tails == other.tails
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.head, self.relation, tuple(self.tails)))

    def to_prompt(self, include_tail: bool = False, **kwargs):
        head = self.head
        relation = self.relation
        tail = self.tails[0] if self.tails else None
        return relation.verbalize(str(head), tail, include_tail=include_tail, **kwargs)

    def to_query(self, decode_method: str = "greedy"):
        if decode_method == "greedy":
            return "{} {}".format(str(self.head), str(self.relation))
        elif decode_method == "beam":
            return "{} {} [GEN]".format(str(self.head), str(self.relation))
        else:
            raise ValueError

    def copy(self):
        return Knowledge(head=self.head, relation=self.relation, tails=self.tails)

    def to_json(self, only_one_tail=False):
        return {
            "head": str(self.head),
            "relation": str(self.relation),
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

    def __add__(self, other):
        return self.union(other)

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    @classmethod
    def from_jsonl(
        cls,
        filepath: str,
        head_attr: str = "head",
        relation_attr: str = "relation",
        tails_attr: str = "tails",
        relation_type: KnowledgeRelationType = KnowledgeRelationType.ATOMIC,
    ):
        kg_list = []

        with open(filepath) as file:
            for line in file:
                kg_json = json.loads(line)
                head = kg_json.get(head_attr)
                relation = KnowledgeRelation.from_text(
                    kg_json.get(relation_attr), relation_type
                )
                tails = kg_json.get(tails_attr)
                kg_list.append(Knowledge(head=head, relation=relation, tails=tails))

        return cls(kg_list)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        header=True,
        head_col: str = "head",
        relation_col: str = "relation",
        tails_col: str = "tails",
        sep=",",
        relation_type: KnowledgeRelationType = KnowledgeRelationType.ATOMIC,
    ):
        kg_list = []
        graph_df = pd.read_csv(
            filepath, sep=sep, header=header, names=[head_col, relation_col, tails_col]
        )

        for _, row in graph_df.iterrows():
            head = row[head_col]
            relation = KnowledgeRelation.from_text(row[relation_col], relation_type)
            tails = row[tails_col]
            kg_list.append(Knowledge(head=head, relation=relation, tails=tails))

        return cls(kg_list)

    def to_jsonl(self, filepath):
        with open(filepath, "w") as file:
            lines = []
            for kg in self.graph:
                lines.append(json.dumps(kg.to_json()))
            file.writelines("\n".join(lines))

    def to_dataframe(self):
        return pd.DataFrame([kg.to_json(only_one_tail=True) for kg in self.graph])

    def union(self, other):
        return KnowledgeGraph(set(self.graph).union(set(other.graph)))

    def intersection(self, other):
        return KnowledgeGraph(set(self.graph).intersection(set(other.graph)))

    def difference(self, other):
        return KnowledgeGraph(set(self.graph).difference(set(other.graph)))
