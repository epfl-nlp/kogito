# <span style="background-color:#dce755">k</span>ogito
A Python NLP Commonsense Reasoning library

## Installation

### Installation with pip
**kogito** can be installed using pip.

```sh
pip install kogito
```

It requires a minimum ``python`` version of ``3.8``.

## Setup
**kogito** uses [spacy](https://spacy.io) under the hood for various text processing purposes, so, a [spacy](https://spacy.io) language package has to be installed before running the inference module.

```sh
python -m spacy download en_core_web_sm
``` 
By default, ``CommonsenseInference`` module uses ``en_core_web_sm`` to initialize ``spacy`` pipeline, but a different language pipeline can be specified as well.

## Quickstart
**kogito** provides an easy interface to interact with commonsense reasoning models such as [COMET](https://arxiv.org/abs/2010.05953) to generate inferences from a text input.
Here is a sample usage of the library where you can initialize an inference module, a custom commonsense reasoning model, and generate a knowledge graph from text on the fly.

```python
from kogito.models.bart.comet import COMETBART
from kogito.inference import CommonsenseInference

# Load pre-trained model from HuggingFace
model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")

# Initialize inference module with a spacy language pipeline
csi = CommonsenseInference(language="en_core_web_sm")

# Run inference
text = "PersonX becomes a great basketball player"
kgraph = csi.infer(text, model)

# Save output knowledge graph to JSON file
kgraph.to_jsonl("kgraph.json")
```

Here is an excerpt from the result of the above code sample:

```json
{"head": "PersonX becomes a great basketball player", "relation": "Causes", "tails": [" PersonX practices every day.", " PersonX plays basketball every day", " PersonX practices every day"]}
{"head": "basketball", "relation": "ObjectUse", "tails": [" play with friends", " play basketball with", " play basketball"]}
{"head": "player", "relation": "CapableOf", "tails": [" play game", " win game", " play football"]}
{"head": "great basketball player", "relation": "HasProperty", "tails": [" good at basketball", " good at sports", " very good"]}
{"head": "become player", "relation": "isAfter", "tails": [" play game", " become coach", " play with"]}
```
This is just one way to generate commonsense inferences and **kogito** offers much more. For complete documentation, check out the [kogito docs](https://kogito.readthedocs.io).

## Development

### Setup
**kogito** uses [Poetry](https://python-poetry.org/) to manage its dependencies. 

Install poetry from the official repository first:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Then run the following command to install package dependencies:
```sh
poetry install
```
