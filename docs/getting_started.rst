===============
Getting Started
===============

Installation
============

Installation with pip
*********************
Kogito can be installed using pip.

.. code-block:: shell

   pip install kogito

It requires a minimum ``python`` version of ``3.8``.

Setup
=====
Kogito uses ``spacy`` under the hood for various text processing purposes, so, a ``spacy`` language package has to be installed before running the inference module.

.. code-block:: shell

   python -m spacy download en_core_web_sm

By default, ``CommonsenseInference`` module uses ``en_core_web_sm`` to initialize ``spacy`` pipeline, but a different language pipeline can be specified as well.

Quickstart
===========
Kogito provides an easy interface to interact with commonsense reasoning models such as ``COMET`` to generate inferences from a text input.
Here is a sample usage of the library where you can initialize an inference module, a custom commonsense reasoning model, and generate a knowledge graph from text on the fly.

.. code-block:: python

    from kogito.models.bart.comet_bart import COMETBART
    from kogito.inference import CommonsenseInference

    # Load pre-trained model from HuggingFace
    model = COMETBART.from_pretrained("mismayil/comet-bart")

    # Initialize inference module with a spacy language pipeline
    csi = CommonsenseInference(language="en_core_web_sm")

    # Run inference
    text = "PersonX becomes a great basketball player"
    kgraph = csi.infer(text, model)

    # Save output knowledge graph to JSON file
    kgraph.to_jsonl("kgraph.json")


Here is an excerpt from the result of the above code sample:

.. code-block:: json

   {"head": "PersonX becomes a great basketball player", "relation": "Causes", "tails": [" PersonX practices every day.", " PersonX plays basketball every day", " PersonX practices every day"]}
   {"head": "basketball", "relation": "ObjectUse", "tails": [" play with friends", " play basketball with", " play basketball"]}
   {"head": "player", "relation": "CapableOf", "tails": [" play game", " win game", " play football"]}
   {"head": "great basketball player", "relation": "HasProperty", "tails": [" good at basketball", " good at sports", " very good"]}
   {"head": "become player", "relation": "isAfter", "tails": [" play game", " become coach", " play with"]}

This is just one way to generate commonsense inferences and ``kogito`` offers much more. For information on more use-cases and a complete API reference, you can check out the `User Guide <https://github.com>`_ and `API Reference <https://github.com>`_ pages.