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
train_graph = KnowledgeGraph.from_csv("data/atomic2020/sample_train.tsv", header=None, sep="\t")
val_graph = KnowledgeGraph.from_csv("data/atomic2020/sample_dev.tsv", header=None, sep="\t")
test_graph = KnowledgeGraph.from_csv("data/atomic2020/sample_test.tsv", header=None, sep="\t")
model.train(train_graph=train_graph, val_graph=val_graph, test_graph=test_graph)
