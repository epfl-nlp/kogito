# ATOMIC 2020 Knowledge Graph

All data `*.tsv` files are formatted as follows:
- each line represents a distinct commonsense tuple
- column 1: head node/concept
- column 2: edge relation (e.g., xWant,xAttr,AtLocation)
- column 3: tail node/concept

`train.tsv`, `dev.tsv`, `test.tsv` correspond to train/dev/test splits.


## Paper
Please cite the following work when using this data:

> Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, Yejin Choi (2021).
> (Comet-) Atomic 2020: On Symbolic and Neural Commonsense Knowledge Graphs.
> AAAI 2021