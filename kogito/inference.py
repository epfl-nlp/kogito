from typing import Union, List
from itertools import product

import spacy

from kogito.core.knowledge import Knowledge, KnowledgeGraph
from kogito.core.head import KnowledgeHead
from kogito.core.processors.head import (
    KnowledgeHeadExtractor,
    SentenceHeadExtractor,
    NounPhraseHeadExtractor,
    VerbPhraseHeadExtractor,
)
from kogito.core.relation import KnowledgeRelation
from kogito.core.processors.relation import (
    GraphBasedRelationMatcher,
    KnowledgeRelationMatcher,
    SimpleRelationMatcher,
)
from kogito.models.base import KnowledgeModel


class CommonsenseInference:
    def __init__(self, language: str = "en_core_web_sm") -> None:
        self.language = language
        self.nlp = spacy.load(language, exclude=["ner"])

        self._head_processors = {
            "sentence_extractor": SentenceHeadExtractor("sentence_extractor", self.nlp),
            "noun_phrase_extractor": NounPhraseHeadExtractor(
                "noun_phrase_extractor", self.nlp
            ),
            "verb_phrase_extractor": VerbPhraseHeadExtractor(
                "verb_phrase_extractor", self.nlp
            ),
        }
        self._relation_processors = {
            "simple_relation_matcher": SimpleRelationMatcher(
                "simple_matcher", self.nlp
            ),
            "graph_relation_matcher": GraphBasedRelationMatcher(
                "graph_matcher", self.nlp
            ),
        }

    @property
    def processors(self):
        return {
            "head": list(self._head_processors.keys()),
            "relation": list(self._relation_processors.keys()),
        }

    def infer(
        self,
        text: str = None,
        model: KnowledgeModel = None,
        heads: List[str] = None,
        model_args: dict = None,
        extract_heads: bool = True,
        match_relations: bool = True,
        relations: List[KnowledgeRelation] = None,
        dry_run: bool = False,
        sample_graph: KnowledgeGraph = None,
    ) -> KnowledgeGraph:
        kg_heads = []
        head_relations = []
        head_texts = set()
        model_args = model_args or {}

        if heads:
            for head in heads:
                head_texts.add(head)
                kg_heads.append(KnowledgeHead(text=head))

        if extract_heads:
            if text:
                print("Extracting heads...")
                for head_proc in self._head_processors.values():
                    extracted_heads = head_proc.extract(text)
                    for head in extracted_heads:
                        head_text = head.text.strip().lower()
                        if head_text not in head_texts:
                            kg_heads.append(head)
                            head_texts.add(head_text)
        else:
            if text and text not in head_texts:
                head_texts.add(text)
                kg_heads.append(KnowledgeHead(text=text))

        if match_relations:
            print("Matching relations...")
            for relation_proc in self._relation_processors.values():
                head_relations.extend(
                    relation_proc.match(kg_heads, relations, sample_graph=sample_graph)
                )
        elif relations:
            if not isinstance(relations, list):
                raise ValueError("Relation subset should be a list")

            head_relations.extend(list(product(kg_heads, relations)))
        else:
            raise ValueError("No relation found to match")

        kg_list = []

        for head_relation in head_relations:
            head, relation = head_relation
            kg_list.append(Knowledge(head=head, relation=relation))

        input_graph = KnowledgeGraph(kg_list)

        if sample_graph:
            input_graph = input_graph + sample_graph

        if dry_run or not model:
            return input_graph

        print("Generating commonsense graph...")
        output_graph = model.generate(input_graph, **model_args)

        return output_graph

    def add_processor(
        self, processor: Union[KnowledgeHeadExtractor, KnowledgeRelationMatcher]
    ) -> None:
        if isinstance(processor, KnowledgeHeadExtractor):
            self._head_processors[processor.name] = processor
            processor.lang = self.nlp
        elif isinstance(processor, KnowledgeRelationMatcher):
            self._relation_processors[processor.name] = processor
            processor.lang = self.nlp
        else:
            raise ValueError("Unknown processor")

    def remove_processor(self, processor_name: str) -> None:
        if processor_name in self._head_processors:
            del self._head_processors[processor_name]
        elif processor_name in self._relation_processors:
            del self._relation_processors[processor_name]
