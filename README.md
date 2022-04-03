# kogito
A Python NLP Commonsense Reasoning library


## Setup
Install poetry:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Setup spacy
```sh
python -m spacy download en_core_web_sm

Install jupyter
```sh
pip install jupyter
```

```
Download ATOMIC2020
```sh
cd examples/relation_modeling
mkdir data
cd data
wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
unzip atomic2020_data-feb2021.zip
```