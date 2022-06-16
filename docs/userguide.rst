==========
User Guide
==========

This guide intends to cover the basic concepts of the library, commonsense inference generation and knowledge model training.


Basics
======

Introduction
************
This library provides an easy and extensible interface to perform **knowledge inference** such as **commonsense reasoning** in natural language.

Wikipedia defines commonsense reasoning as *a human-like ability to make presumptions about the type and essence of ordinary situations humans encounter every day*.
It operates on commonsense knowledge which is often implicit and includes judgments about the nature of physical objects, taxonomic properties and people's intentions.
While humans are very good at this task, it has largely remained as one of the long-standing challenges for machines.

However, recent large-scale language models have brought a tremendous progress in natural language processing and new so-called **knowledge models** have emerged that have
exhibited promising results on commonsense reasoning tasks. These models are trained on large **knowledge graphs** composed of **knowledge** tuples that represent the commonsense
knowledge about the world. In this sense, a **knowledge** is represented as a tuple of **(head, relation, tails)** where **head** refers to the source knowledge such as ``PersonX buys lunch``, **relation** to the reasoning
question asked about the source knowledge such as ``What does PersonX need?`` and **tails** to the target knowledge presented as possible answers to this question such as ``bring a wallet``.
**kogito** at its core, provides an intuitive representation and manipulation of these concepts to enable a standardized high level interface for reasoning tasks.

Knowledge
*********
Knowledge in **kogito** is represented using :class:`kogito.core.knowledge.Knowledge` class. It is initialized with an instance of a :class:`kogito.core.head.KnowledgeHead`, a :class:`kogito.core.relation.KnowledgeRelation` and a list of knowledge tails represented
as literal strings. While **heads** and **tails** can be any abritrary text, **relations** are rather pre-defined notions and are based on `ATOMIC <https://allenai.org/data/atomic-2020>`_ and `CONCEPTNET <https://conceptnet.io/>`_ relations. For example, the relation mentioned above
is represented with the ``X_NEED`` object. However, this being said **kogito** also provides a way to define new custom relations and use them to perform commonsense reasoning. Please, refer to the 
`Custom Relations`_ section for more on this.

.. code-block:: python
   
   from kogito.core.head import KnowledgeHead
   from kogito.core.knowledge import Knowledge
   from kogito.core.relation import X_NEED

   head = KnowledgeHead("PersonX buys lunch")
   knowledge = Knowledge(head=head, relation=X_NEED, tails=["bring a wallet"])

Moreover, all 3 classes allow a direct comparison between their respective instances based on the text and type of the instance. Two ``KnowledgeHead`` objects are considered equal
if their respective underlying texts match, two ``KnowledgeRelation`` objects are equal if their representative texts and types (relation type refers to the knowledge base it comes from e.g. ATOMIC) and
finally, two ``Knowledge`` objects are equal when their heads, relations and tails match.

Knowledge Graph
***************
Knowledge graph in **kogito** is represented using :class:`kogito.core.knowledge.KnowledgeGraph` class and is simply a collection of ``Knowledge`` objects. This class provides an easy interface to read, manipulate and write
knowledge instances.

.. code-block:: python

   from kogito.core.knowledge import Knowledge, KnowledgeGraph
   from kogito.core.head import KnowledgeHead
   from kogito.core.relation import X_NEED, CAUSES

   knowledge1 = Knowledge(head=KnowledgeHead("PersonX buys lunch"), relation=X_NEED, tails=["bring a wallet"])
   knowledge2 = Knowledge(head=KnowledgeHead("Throwing a party"), relation=CAUSES, tails=["have fun"])

   kgraph = KnowledgeGraph([knowledge1, knowledge2])

``KnowledgeGraph`` can be easily iterated over similar to Python sequences.

.. code-block:: python

   for knowledge in kgraph:
       print(knowledge)

``KnowledgeGraph`` can be initialized from different input sources such as *csv*, *json* etc.
Given following csv and jsonl files:

.. admonition:: sample_graph.csv

   PersonX buys lunch | xNeed | bring a wallet

.. admonition:: sample_graph.jsonl

   {"source": "PersonX buys lunch", "rel": "xNeed", "tails": ["bring a wallet"]}

   {"source": "Throwing a party", "rel": "Causes", "tails": ["have fun"]}

we can instantiate knowledge graphs as below:

.. code-block:: python

   # From csv
   kgraph1 = KnowledgeGraph.from_csv("sample_graph1.csv", sep="|", header=None)

   # From jsonl (list of json objects)
   kgraph2 = KnowledgeGraph.from_jsonl("sample_graph2.jsonl", head_attr="source", relation_attr="rel", tails_attr="targets")


**kogito** also provides an out-of-box set-like capabilities for ``KnowledgeGraph`` instances such as **union** (also with overloaded **+** and **|**), 
**intersection** (also with overloaded **&**) and **difference** (also with overloaded **-**) operators.

.. code-block:: python
   
   # Union
   kgraph3 = kgraph1 + kgraph2 # kgraph1.union(kgraph2)

   # Intersection
   kgraph3 = kgraph1 & kgraph2 # kgraph1.intersection(kgraph2)

   # Difference
   kgraph3 = kgraph1 - kgraph2 # kgraph1.difference(kgraph2)

``KnowledgeGraph`` object can also be written to different output formats.

.. code-block:: python

   kgraph1.to_jsonl("sample_graph1.jsonl")


Knowledge Model
***************
Base knowledge model in **kogito** is represented by the :class:`kogito.core.model.KnowledgeModel` class and provides an abstract interface to be implemented by concrete model instances.
More specifically, these following methods, namely, ``train``, ``evaluate``, ``generate``, ``from_pretrained`` and ``save_pretrained`` are defined and allow for training, evaluating, querying (generating inferences from),
loading and saving models respectively. For inference generation, these models take an instance of ``KnowledgeGraph`` (generally this graph will be incomplete i.e. each knowledge instance in its collection will be missing **tails** since we want to predict those)
and output a complete version of the input graph (**tails** filled in).
For more information on specific models available as part of **kogito**, please refer to the `Models`_ section.
Here is an example of loading a pre-trained model from `HuggingFace <https://huggingface.co/>`_.

.. code-block:: python

    from kogito.models.bart.comet import COMETBART

    # Load pre-trained model from HuggingFace
    model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")


Inference
=========
**kogito** offers a simple, yet powerful commonsense inference module called :class:`kogito.inference.CommonsenseInference`. It is initialized with a (`spacy <https://spacy.io>`_) language of choice (by default, ``en_core_web_sm``).
Then its ``infer`` method can be called with various arguments to generate commonsense inferences. Here we will walk through some common use-cases for this module and for complete API reference,
you can refer to `API Reference <https://kogito.readthedocs.io/en/latest/api.html>`_.

.. code-block:: python

    from kogito.inference import CommonsenseInference

    # Initialize inference module with a spacy language pipeline
    csi = CommonsenseInference(language="en_core_web_sm")

Head Extraction
***************
As mentioned before, knowledge models take as input a knowledge graph composed of knowledge tuples, but **kogito** in addition to this offers a way to automatically extract relevant knowledge heads
from the input text to feed into these models. 

.. code-block:: python

    text = "PersonX becomes a great basketball player"
    kgraph = csi.infer(text, model)

Under the hood, **kogito** applies various head extraction methods to the given text. By default, following extraction methods are applied automatically:

- Sentence Extraction (:class:`kogito.core.processors.head.SentenceHeadExtractor`)

  Extracts sentences from text.

- Noun Phrase Extraction (:class:`kogito.core.processors.head.NounPhraseHeadExtractor`)

  Extracts noun phrases from text.

- Verb Phrase Extraction (:class:`kogito.core.processors.head.VerbPhraseHeadExtractor`)

  Extracts verb phrases from text.

You can list all default head extractors as below:

.. code-block:: python

   print(csi.processors)

which will output (it also outputs relation matchers which will be explained in the next section):

.. code-block:: json

   {
      "head": ["sentence_extractor", "noun_phrase_extractor", "verb_phrase_extractor"],
      "relation": ["simple_relation_matcher", "graph_relation_matcher"]
   }

You can also optionally remove head extractors by their name:

.. code-block:: python

   csi.remove_processor("noun_phrase_extractor")

**kogito** also allows you to define your own head extractors. For this, you simply need to implement the :class:`kogito.core.processors.head.KnowledgeHeadExtractor` interface and register the new extractor with the 
inference module. Here is one example that extracts only adjectives from the text: 

.. code-block:: python

   from typing import Optional, List
   from spacy.tokens import Doc
   import spacy

   from kogito.core.processors.head import KnowledgeHeadExtractor, KnowledgeHead

   class AdjectiveHeadExtractor(KnowledgeHeadExtractor):
      def extract(self, text: str, doc: Optional[Doc] = None) -> List[KnowledgeHead]:
         if not doc:
               doc = self.lang(text)

         heads = []

         for token in doc:
               if token.pos_ == "ADJ":
                  heads.append(KnowledgeHead(text=token.text, entity=token))
         
         return heads

   adj_extractor = AdjectiveHeadExtractor("adj_extractor", spacy.load("en_core_web_sm"))
   csi.add_processor(adj_extractor)


Relation Matching
*****************
Of course, knowledge heads are not enough on their own to query knowledge models, we also need to supply the knowledge relations, in other words the questions we want to ask about the knowledge heads.
Luckily, **kogito** also provides an ability to automatically match relevant relations to the extracted heads.
By default, following relation matching methods are applied:

- Simple Heuristics-based Relation Matching  (:class:`kogito.core.processors.relation.SimpleRelationMatcher`)

  Matches heads based on their syntactic category (noun phrase, verb phrase etc.)

- Graph-based Relation Matching (:class:`kogito.core.processors.relation.GraphBasedRelationMatcher`)

  Matches heads to relations provided in a sample graph (for more info on this, see `Custom Relations`_)

and following model-based relation matchers are available out-of-the-box to be added. These models have been trained as a classifier to match heads to one or more of the relation categories of `ATOMIC <https://allenai.org/data/atomic-2020>`_, namely, 
:data:`kogito.core.relation.PHYSICAL_RELATIONS`, :data:`kogito.core.relation.EVENT_RELATIONS` and :data:`kogito.core.relation.SOCIAL_RELATIONS`.

- Simple Word Embedding model based matcher (:class:`kogito.core.processors.relation.SWEMRelationMatcher`)
- DistilBert model based matcher (:class:`kogito.core.processors.relation.DistilBertRelationMatcher`)
- Bert model based matcher (:class:`kogito.core.processors.relation.BertRelationMatcher`)

These matchers can simply be added to the inference module as below:

.. code-block:: python

   from kogito.core.processors.relation import SWEMRelationMatcher

   csi.add_processor(SWEMRelationMatcher())

Similar to head extraction, relation matching methods can also be optionally removed:

.. code-block:: python

   csi.remove_processor("simple_relation_matcher")

and custom ones can be added. Here is an example where each head is matched with the same 2 relations:

.. code-block:: python

   from typing import List, Tuple

   from kogito.core.processors.head import KnowledgeHead
   from kogito.core.processors.relation import KnowledgeRelationMatcher
   from kogito.core.relation import KnowledgeRelation, X_NEED, CAUSES

   class ConstantRelationMatcher(KnowledgeRelationMatcher):
      def match(
         self, heads: List[KnowledgeHead], relations: List[KnowledgeRelation] = None, **kwargs
      ) -> List[Tuple[KnowledgeHead, KnowledgeRelation]]:
         head_relations = []

         for head in heads:
               head_relations.append((head, X_NEED))
               head_relations.append((head, CAUSES))

         return head_relations
   
   const_rel_matcher = ConstantRelationMatcher("const_rel_matcher", spacy.load("en_core_web_sm"))
   csi.add_processor(const_rel_matcher)


Manual Mode
***********
Beyond automatic head extraction and relation matching, **kogito** also provides several manual controls. 
For example, you can specify additional heads manually as a list (either as a text or a ``KnowledgeHead`` instance). 

.. code-block:: python
   
   text = "PersonX becomes a great basketball player"
   heads = ["tennis player", "athlete"]
   kgraph = csi.infer(text=text, heads=heads, model=model)

or completely switch off head extraction by either omitting the text or setting ``extract_heads`` flag to ``False``.
In case a text is provided with the flag switched off, text is taken to be head as is and no head extraction is applied.

.. code-block:: python
   
   text = "PersonX becomes a great basketball player"
   heads = ["tennis player", "athlete"]
   kgraph = csi.infer(text=text, heads=heads, extract_heads=False, model=model)

Similarly, you can specify a subset of relations to match from. Here relation matching will still be performed, but only from the list provided.

.. code-block:: python
   
   from kogito.core.relation import PHYSICAL_RELATIONS

   heads = ["tennis player", "athlete"]
   kgraph = csi.infer(heads=heads, relations=PHYSICAL_RELATIONS, model=model)

or alternatively, you can switch off automatic smart relation matching by setting ``match_relations`` flag to ``False`` which will result in heads being matched with all the relations provided.

.. code-block:: python
   
   from kogito.core.relation import PHYSICAL_RELATIONS

   heads = ["tennis player", "athlete"]
   kgraph = csi.infer(heads=heads, relations=PHYSICAL_RELATIONS, match_relations=False, model=model)

Dry-run Mode
************
If you just want to see the results of head extraction and relation matching without querying the model for actual results, you can do so by either omitting ``model`` argument or
by setting ``dry_run`` flag to ``True``.

.. code-block:: python

   kgraph = csi.infer(text="PersonX becomes a great basketball player", model=model, dry_run=True)
   kgraph.to_jsonl("kgraph.json")

which will output an incomplete knowledge graph (i.e. without tails) like below:

.. code-block:: json

   {"head": "PersonX becomes a great basketball player", "relation": "Causes", "tails": []}
   {"head": "basketball", "relation": "ObjectUse", "tails": []}
   {"head": "player", "relation": "CapableOf", "tails": []}
   {"head": "great basketball player", "relation": "HasProperty", "tails": []}
   {"head": "become player", "relation": "isAfter", "tails": []}

Custom Relations
****************
As mentioned before, knowledge relations are rather fixed, pre-defined notions based on `ATOMIC <https://allenai.org/data/atomic-2020>`_ and `CONCEPTNET <https://conceptnet.io/>`_ knowledge bases. However, one might want to define their own custom relations
and perform commonsense reasoning based on these new relations. **kogito** also provides this capability through large language models such as GPT-3. 
In order to do this, we need to use :class:`kogito.models.gpt3.zeroshot.GPT3Zeroshot` model, define and register our new relation using ``KnowledgeRelation`` class and construct a sample knowledge graph with examples for our new relations.

To define our new relation, we need to provide a ``verbalizer`` function to convert the knowledge tuple into a meaningful sentence in natural language and a ``prompt`` text that explains the new relation
as an instruction (these are required to interact with the GPT-3 model). Let's define a new relation called ``X_WISHES`` which does not exist in any of the knowledge bases.

.. code-block:: python

   from kogito.core.relation import KnowledgeRelation, register_relation

   def x_wishes_verbalizer(head, **kwargs):
      # index will be passed from the model
      # so that we can enumerate samples which helps with inference
      index = kwargs.get("index")
      index_txt = f"{index}" if index is not None else ""
      return f"Situation {index_txt}: {head}\nWishes: As a result, PersonX wishes"

   X_WISHES = KnowledgeRelation("xWishes",
                                verbalizer=x_wishes_verbalizer,
                                prompt="How does this situation affect each character's wishes?")
   register_relation(X_WISHES)

Then we construct the following sample graph showing examples for our new relation.

.. admonition:: sample_graph.csv

   PersonX is at a party  |  xWishes	| to drink beer and dance

   PersonX bleeds a lot	 |  xWishes |	to see a doctor

   PersonX works as a cashier	 |  xWishes	| to be a store manager

   PersonX gets dirty	|  xWishes	| to clean up

   PersonX stays up all night studying	 |  xWishes	| to sleep all day

   PersonX gets PersonY's autograph	|  xWishes	| to have a relationship with PersonY

   PersonX ends a friendship	|  xWishes	| to meet new people

   PersonX makes his own costume	|  xWishes	| to go to a costume party

   PersonX calls PersonY	|  xWishes	| to have a long chat

   PersonX tells PersonY a secret	|  xWishes	| to get PersonY's advice

   PersonX mows the lawn	|  xWishes	| to get a new lawnmower

Note that the unique relation name provided above in the definition (i.e. "xWishes") should match the one in the examples.

Finally, we initialize our GPT-3 model and run the inference:

.. code-block:: python

   from kogito.inference import CommonsenseInference
   from kogito.core.knowledge import KnowledgeGraph
   from kogito.models.gpt3.zeroshot import GPT3Zeroshot

   csi = CommonsenseInference()
   # Here we remove the simple relation matcher for simplicity
   csi.remove_processor("simple_relation_matcher")

   # Initialize GPT-3 model using API access
   model = GPT3Zeroshot(api_key="<your GPT-3 API Key>", model_name="text-davinci-002")

   sample_graph = KnowledgeGraph.from_csv("sample_graph.csv", sep="|", header=None)

   heads = ["PersonX makes a huge mistake", "PersonX sees PersonY's point"]

   kgraph = csi.infer(model=model, heads=heads, sample_graph=sample_graph)

Models
======
**kogito** offers following knowledge models for inference:

- ``COMETBART`` (:class:`kogito.models.bart.comet.COMETBART`)
- ``COMETGPT2`` (:class:`kogito.models.gpt2.comet.COMETGPT2`)
- ``GPT2Zeroshot`` (:class:`kogito.models.gpt2.zeroshot.GPT2Zeroshot`)
- ``GPT3Zeroshot`` (:class:`kogito.models.gpt3.zeroshot.GPT3Zeroshot`)

All of these models implement the ``KnowledgeModel`` interface which provides these main methods to interact with these models: ``train``, ``generate``, ``evaluate``, ``save_pretrained`` and ``from_pretrained``.

Inference
*********
``generate`` method is used to make inferences with knowledge models. It takes an (incomplete i.e. without tails) input knowledge graph and outputs a (completed i.e. tails generated) knowledge graph.

Given an input graph in a *json* format like below:

.. admonition:: input_graph.jsonl

   {"relation": "xNeed", "head": "PersonX takes things for granted", "tails": []}

   {"relation": "xWant", "head": "PersonX pleases ___ to make", "tails": []}

   {"relation": "xEffect", "head": "PersonX shoves PersonY back", "tails": []}

   {"relation": "isAfter", "head": "PersonX wants to go", "tails": []}

   {"relation": "xEffect", "head": "PersonX hits by lightning", "tails": []}

   {"relation": "xNeed", "head": "PersonX finally meet PersonY", "tails": []}

   {"relation": "ObjectUse", "head": "chain", "tails": []}

We can generate inferences for example using ``COMETBART`` model as below:

.. code-block:: python

   from kogito.core.knowledge import KnowledgeGraph
   from kogito.models.bart.comet import COMETBART

   input_graph = KnowledgeGraph.from_jsonl("input_graph.jsonl")

   # Load a model from HuggingFace
   model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
   output_graph = model.generate(input_graph)
   output_graph.to_jsonl("output_graph.jsonl")

While COMET based models have been trained specifically on knowledge graphs, zeroshot models are based on the publicly available language models.
``GPT2Zeroshot`` model by default uses the publicly available `gpt2 <https://huggingface.co/gpt2>`_ model from HuggingFace and can simply be initialized using the class constructor:

.. code-block:: python

   from kogito.models.gpt2.zeroshot import GPT2Zeroshot

   model = GPT2Zeroshot()

``GPT3Zeroshot`` model on the other hand is currently only available through public API access, hence, an API key is required to interact with this model.

.. code-block:: python

   from kogito.models.gpt3.zeroshot import GPT3Zeroshot

   model = GPT3Zeroshot(api_key="<your API key>", model_name="text-davince-002")

Training
********
COMET models have been trained based on the paper `COMET-ATOMIC2020: On Symbolic and Neural Commonsense Knowledge Graphs <https://arxiv.org/abs/2010.05953>`_ and made available as pre-trained models through HuggingFace:

.. code-block:: python

   from kogito.models.bart.comet import COMETBART
   from kogito.models.gpt2.comet import COMETGPT2

   comet_bart = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
   comet_gpt2 = COMETGPT2.from_pretrained("mismayil/comet-gpt2-ai2")

However, if you wish to train these models on a new dataset and/or with different hyperparameters, you can do so using the provided ``train`` method. This method takes a training dataset as an instance of a ``KnowledgeGraph`` and additional hyperparameters depending on the model type.
Please, refer to the `API Reference <https://kogito.readthedocs.io/en/latest/api.html>`_ for more details on specific parameters accepted by this method for each model.

For example, here is a sample code to train a  ``COMETBART`` model:

.. code-block:: python

   from kogito.core.knowledge import KnowledgeGraph
   from kogito.models.bart.comet import COMETBART, COMETBARTConfig


   config = COMETBARTConfig(
      output_dir="bart",
      num_workers=2,
      learning_rate=1e-5,
      gpus=1,
      sortish_sampler=True,
      atomic=True,
      pretrained_model="facebook/bart-large",
   )
   model = COMETBART(config)
   train_graph = KnowledgeGraph.from_csv("train.tsv")
   val_graph = KnowledgeGraph.from_csv("val.tsv")
   test_graph = KnowledgeGraph.from_csv("test.tsv")

   model.train(train_graph=train_graph, val_graph=val_graph, test_graph=test_graph)

   # Save as a pretrained model
   model.save_pretrained("comet-bart/v1")


Evaluation
**********
Knowledge models can also be evaluated on various metrics. ``KnowledgeModel.evaluate`` method takes an input knowledge graph (complete with reference tails), runs a generation on it and then
computes various scores based on the reference and generation tails and outputs a dictionary of these scores. You can also specify how many generations to consider for evaluation and 
pass any extra arguments required for model generation.

Following metrics are available and enabled by default: 

- `BLEU <https://en.wikipedia.org/wiki/BLEU>`_ (``"bleu"``)
- `ROUGE <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_ (``"rouge"``)
- `CIDEr <https://arxiv.org/abs/1411.5726>`_ (``"cider"``)
- `METEOR <https://en.wikipedia.org/wiki/METEOR>`_ (``"meteor"``)
- `BERTScore <https://arxiv.org/abs/1904.09675>`_ (``"bert-score"``)

Here is an example of evaluating ``COMETBART`` model with some metrics:

.. code-block:: python

   from kogito.core.knowledge import KnowledgeGraph
   from kogito.models.bart.comet import COMETBART

   input_graph = KnowledgeGraph.from_jsonl("test_atomic2020_sample.json")

   model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
   # Here batch_size is an extra parameter for model generation
   scores: dict = model.evaluate(input_graph, metrics=["bleu", "rouge"], top_k=2, batch_size=256)

   print(scores)