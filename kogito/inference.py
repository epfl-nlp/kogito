from typing import Union

import spacy

from kogito.core.knowledge import (
    Knowledge,
    KnowledgeBase,
    KnowledgeGraph,
    CONCEPTNET_RELATIONS,
)
from kogito.core.head import (
    KnowledgeHeadExtractor,
    SentenceHeadExtractor,
    PhraseHeadExtractor,
    NounHeadExtractor,
)
from kogito.core.relation import KnowledgeRelationMatcher, SimpleRelationMatcher
from kogito.models.base import KnowledgeModel


class CommonsenseInference:
    def __init__(self, language: str = "en_core_web_sm") -> None:
        self.language = language
        self.nlp = spacy.load(language)

        self._head_processors = {
            "sentence_extractor": SentenceHeadExtractor("sentence_extractor", self.nlp),
            "phrase_extractor": PhraseHeadExtractor("phrase_extractor", self.nlp),
            "noun_extractor": NounHeadExtractor("noun_extractor", self.nlp),
        }
        self._relation_processors = {
            "simple_matcher": SimpleRelationMatcher("simple_matcher")
        }

    @property
    def processors(self):
        return {
            "head": list(self._head_processors.keys()),
            "relation": list(self._relation_processors.keys()),
        }

    def infer(
        self, text: str, model: KnowledgeModel, model_args: dict = None
    ) -> KnowledgeGraph:
        heads = []
        head_relations = []
        head_texts = set()
        model_args = model_args or {}

        print("Extracting heads...")
        for head_proc in self._head_processors.values():
            extracted_heads = head_proc.extract(text)
            for head in extracted_heads:
                head_text = head.text.strip().lower()
                if head_text not in head_texts:
                    heads.append(head)
                    head_texts.add(head_text)

        print("Matching relations...")
        for relation_proc in self._relation_processors.values():
            head_relations.extend(relation_proc.match(heads))

        kg_list = []

        for head_relation in head_relations:
            head, relation = head_relation
            kg_base = (
                KnowledgeBase.CONCEPTNET
                if relation in CONCEPTNET_RELATIONS
                else KnowledgeBase.ATOMIC
            )
            kg_list.append(Knowledge(head=head.text, relation=relation, base=kg_base))

        input_graph = KnowledgeGraph(kg_list)
        print("Generating commonsense graph...")
        output_graph = model.generate(input_graph, **model_args)

        return output_graph

    def add_processor(
        self, processor: Union[KnowledgeHeadExtractor, KnowledgeRelationMatcher]
    ) -> None:
        if isinstance(processor, KnowledgeHeadExtractor):
            self._head_processors[processor.name] = processor
        elif isinstance(processor, KnowledgeRelationMatcher):
            self._relation_processors[processor.name] = processor
        else:
            raise ValueError("Unknown processor")

    def remove_processor(self, processor_name: str) -> None:
        if processor_name in self._head_processors:
            del self._head_processors[processor_name]
        elif processor_name in self._relation_processors:
            del self._relation_processors[processor_name]
