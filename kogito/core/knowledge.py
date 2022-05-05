from typing import List, Union, Optional
import pandas as pd
import json

from kogito.core.relation import KnowledgeRelation, KnowledgeRelationType
from kogito.core.head import KnowledgeHead

EOS_TOKEN = "[EOS]"
GEN_TOKEN = "[GEN]"
PAD_TOKEN = "[PAD]"

DECODE_METHODS = ["greedy", "beam"]


class Knowledge:
    """
    Represents a concept of Knowledge
    """

    def __init__(
        self,
        head: Optional[Union[KnowledgeHead, str]] = None,
        relation: Optional[Union[KnowledgeRelation, str]] = None,
        tails: Optional[List[str]] = None,
    ) -> None:
        """Initialize a Knowledge instance.

        Args:
            head (Optional[Union[KnowledgeHead, str]], optional): Instance of a knowledge head. Defaults to None.
            relation (Optional[Union[KnowledgeRelation, str]], optional): Instance of a knowledge relation.
                                                                         Defaults to None.
            tails (Optional[List[str]], optional): List of knowledge tails. Defaults to None.
        """
        self.head = head if isinstance(head, KnowledgeHead) else KnowledgeHead(head)
        self.relation = (
            relation
            if isinstance(relation, KnowledgeRelation)
            else KnowledgeRelation.from_text(relation)
        )
        self.tails = tails or []
        if not isinstance(self.tails, (list, tuple)):
            self.tails = [str(self.tails)]

    def __repr__(self) -> str:
        return f'Knowledge(head="{str(self.head)}", relation="{str(self.relation)}", tails={self.tails})'

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Knowledge)
            and self.head == other.head
            and self.relation == other.relation
            and self.tails == other.tails
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.head, self.relation, tuple(self.tails)))

    def to_prompt(self, include_tail: bool = False, **kwargs) -> str:
        """Convert knowledge to a prompt text.

        Args:
            include_tail (bool, optional): Whether to include tails in the prompt. Defaults to False.

        Returns:
            str: Prompt text for the knowledge
        """
        head = self.head
        relation = self.relation
        tail = self.tails[0] if self.tails else None
        return relation.verbalize(str(head), tail, include_tail=include_tail, **kwargs)

    def to_query(self, decode_method: str = "greedy") -> str:
        """Convert knowledge to a query text

        Args:
            decode_method (str, optional): Decoding method. Defaults to "greedy".

        Raises:
            ValueError: When decode_method is not recognized.

        Returns:
            str: Query text for the knowledge
        """
        if decode_method == "greedy":
            return "{} {}".format(str(self.head), str(self.relation))
        elif decode_method == "beam":
            return "{} {} [GEN]".format(str(self.head), str(self.relation))
        else:
            raise ValueError

    def copy(self) -> "Knowledge":
        """Copy itself.

        Returns:
            Knowledge: Copied knowledge
        """
        return Knowledge(head=self.head, relation=self.relation, tails=self.tails)

    def to_json(self, only_one_tail: bool = False) -> dict:
        """Convert knowledge to dictionary

        Args:
            only_one_tail (bool, optional): Include only one tail. Defaults to False.

        Returns:
            dict: Jsonified knowledge
        """
        return {
            "head": str(self.head),
            "relation": str(self.relation),
            "tails": self.tails[0] if self.tails and only_one_tail else self.tails,
        }


class KnowledgeGraph:
    """
    Represents a concept of Knowledge Graph.
    """

    def __init__(self, graph: List[Knowledge]) -> None:
        """Initialize a knowledge graph

        Args:
            graph (List[Knowledge]): List of Knowledge instances
        """
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
    ) -> "KnowledgeGraph":
        """Initialize a knowledge graph from json file.

        Args:
            filepath (str): Path to the graph file.
            head_attr (str, optional): JSON attribute for head. Defaults to "head".
            relation_attr (str, optional): JSON attribute for relation. Defaults to "relation".
            tails_attr (str, optional): JSON attribute for tails. Defaults to "tails".
            relation_type (KnowledgeRelationType, optional): Type of relation to use.
                                                            Defaults to KnowledgeRelationType.ATOMIC.

        Returns:
            KnowledgeGraph: An instance of KnowledgeGraph
        """
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
        header: bool = True,
        head_col: str = "head",
        relation_col: str = "relation",
        tails_col: str = "tails",
        sep: str = ",",
        relation_type: KnowledgeRelationType = KnowledgeRelationType.ATOMIC,
    ) -> "KnowledgeGraph":
        """Initialize a knowledge graph from csv file.

        Args:
            filepath (str): Path to the graph file.
            header (bool, optional): Whether to look for header. Defaults to True.
            head_col (str, optional): Head column name. Defaults to "head".
            relation_col (str, optional): Relation column name. Defaults to "relation".
            tails_col (str, optional): Tails column name. Defaults to "tails".
            sep (str, optional): Delimiter to use. Defaults to ",".
            relation_type (KnowledgeRelationType, optional): Relation type to use.
                                                            Defaults to KnowledgeRelationType.ATOMIC.

        Returns:
            KnowledgeGraph: An instance of KnowledgeGraph
        """
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

    def to_jsonl(self, filepath: str) -> None:
        """Write knowledge graph to a json file

        Args:
            filepath (str): JSON file path
        """
        with open(filepath, "w") as file:
            lines = []
            for kg in self.graph:
                lines.append(json.dumps(kg.to_json()))
            file.writelines("\n".join(lines))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert knowledge graph to a pandas dataframe

        Returns:
            pd.DataFrame: Pandas dataframe of a knowledge graph
        """
        return pd.DataFrame([kg.to_json(only_one_tail=True) for kg in self.graph])

    def union(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Union two knowledge graphs

        Args:
            other (KnowledgeGraph): Knowledge graph to union with.

        Returns:
            KnowledgeGraph: Merged knowledge graph
        """
        return KnowledgeGraph(set(self.graph).union(set(other.graph)))

    def intersection(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Intersect two knowledge graphs

        Args:
            other (KnowledgeGraph): Knowledge graph to intersect with.

        Returns:
            KnowledgeGraph: Intersection of two graphs
        """
        return KnowledgeGraph(set(self.graph).intersection(set(other.graph)))

    def difference(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Subtract knowledge graphs

        Args:
            other (KnowledgeGraph): Knowledge graph to subtract.

        Returns:
            KnowledgeGraph: Difference of two graphs
        """
        return KnowledgeGraph(set(self.graph).difference(set(other.graph)))
