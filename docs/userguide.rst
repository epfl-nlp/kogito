==========
User Guide
==========

This guide intends to cover the basic concepts of the library, commonsense inference generation and knowledge model training.


Basics
======

This library provides an easy and extensible interface to perform **commonsense reasoning** in natural language.

Reasoning
*********************
Wikipedia defines commonsense reasoning as *a human-like ability to make presumptions about the type and essence of ordinary situations humans encounter every day*.
It operates on commonsense knowledge which is often implicit and includes judgments about the nature of physical objects, taxonomic properties and people's intentions.
While humans are very good at this task, it has largely remained as one of the long-standing challenges for machines.

However, recent large-scale language models have brought a tremendous progress in natural language processing and new so-called **knowledge models** have emerged that have
exhibited promising results on commonsense reasoning tasks. These models are trained on large **knowledge graphs** composed of **knowledge** tuples that represent the commonsense
knowledge about the world. In this sense, a **knowledge** is represented as a tuple of **(head, relation, tails)** where **head** refers to the source knowledge such as ``PersonX buys lunch``, **relation** to the reasoning
question asked about the source knowledge such as ``what does PersonX need?`` and **tails** to the target knowledge presented as possible answers to this question such as ``bring a wallet``.
**kogito** at its core, provides an intuitive representation and manipulation of these concepts to enable a standardized high level interface for reasoning tasks.

Knowledge
*********
Knowledge in **kogito** is represented using ``Knowledge`` class. It is initialized with an instance of a ``KnowledgeHead``, a ``KnowledgeRelation`` and a list of knowledge tails represented
as literal strings. While **heads** and **tails** can be any abritrary text, **relations** are rather pre-defined notions and are based on **ATOMIC** and **CONCEPTNET** relations. For example, the relation mentioned above
is represented with the ``X_NEED`` object. However, this being said **kogito** also provides a way to define new custom relations and use them to perform commonsense reasoning. Please, refer to the 
**Custom Relations** section for more on this.

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
Knowledge graph in **kogito** is represented using ``KnowledgeGraph`` class and is simply a collection of ``Knowledge`` objects. This class provides an easy interface to read, manipulate and write
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


**kogito** also provides an out-of-box set-like capabilities for ``KnowledgeGraph`` instances such as *union* (also with overloaded **+** and **|**), 
*intersection* (also with overloaded **&**) and *difference* (also with overloaded **-**) operators.

.. code-block:: python
   
   # Union
   kgraph3 = kgraph1 + kgraph2 # kgraph1.union(kgraph2)

   # Intersection
   kgraph3 = kgraph1 & kgraph2 # kgraph1.intersection(kgraph2)

   # Difference
   kgraph3 = kgraph1 - kgraph2 # kgraph1.difference(kgraph2)

``KnowledgeGraph`` object can also be written to different output formats.

.. code-block:: python

   kgraph1.to_jsonl("sample_graph3.jsonl")


Knowledge Model
***************
Base knowledge model in **kogito** is represented by the ``KnowledgeModel`` class and provides an abstract interface to be implemented by concrete model instances.
More specifically, 4 abstract methods, namely, ``train``, ``generate``, ``from_pretrained`` and ``save_pretrained`` are defined and allow for training, querying (generating inferences from),
loading and saving models respectively. For more information on specific models available as part of **kogito**, please refer to the **Models** section.

Inference
=========


Models
========