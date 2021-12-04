#!/bin/bash
python -m spacy download en
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"

mkdir -p data
(
    cd data

    # Download ATOMIC
    wget https://storage.googleapis.com/ai2-mosaic/public/atomic/v1.0/atomic_data.tgz
    mkdir -p atomic
    tar -C ./atomic -xzf atomic_data.tgz

    # Download ATOMIC2020
    wget https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip
    unzip atomic2020_data-feb2021.zip
    mv atomic2020_data-feb2021 atomic2020
)

# Prepare atomic2020 for COMET-GPT-2 training
python scripts/convert_atomic.py data/atomic2020/test.tsv data/atomic2020/atomic_test.tsv
python scripts/convert_atomic.py data/atomic2020/train.tsv data/atomic2020/atomic_train.tsv
python scripts/convert_atomic.py data/atomic2020/dev.tsv data/atomic2020/atomic_dev.tsv

# Prepare atomic2020 for COMET-BART training
python scripts/convert_atomic_bart.py data/atomic2020/train.tsv data/atomic2020/train.source data/atomic2020/train.target
python scripts/convert_atomic_bart.py data/atomic2020/test.tsv data/atomic2020/test.source data/atomic2020/test.target
python scripts/convert_atomic_bart.py data/atomic2020/dev.tsv data/atomic2020/val.source data/atomic2020/val.target

# Prepare atomic2020 test file
python scripts/convert_atomic_test.py system_eval/test.tsv data/atomic2020/test_atomic2020.jsonl