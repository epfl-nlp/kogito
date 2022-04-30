import csv
from kogito.core.knowledge import Knowledge, KnowledgeGraph
from kogito.models.bart.comet import COMETBART, COMETBARTConfig


def kg_graph_from_tsv(filepath):
    with open(filepath) as file:
        reader = csv.DictReader(
            file,
            delimiter="\t",
            fieldnames=["head", "relation", "tail", "id1", "id2", "score"],
        )
        graph = []
        for row in reader:
            graph.append(
                Knowledge(
                    head=row["head"], relation=row["relation"], tails=[row["tail"]]
                )
            )

        return KnowledgeGraph(graph)


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
train_graph = kg_graph_from_tsv("data/atomic2020/sample_train.tsv")
val_graph = kg_graph_from_tsv("data/atomic2020/sample_dev.tsv")
test_graph = kg_graph_from_tsv("data/atomic2020/sample_test.tsv")
model.train(train_graph=train_graph, val_graph=val_graph, test_graph=test_graph)
