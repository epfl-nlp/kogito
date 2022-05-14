from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART, COMETBARTConfig


config = COMETBARTConfig(
    output_dir="models/comet-bart",
    task="summarization",
    n_val=100,
    num_workers=2,
    learning_rate=1e-5,
    gpus=1,
    sortish_sampler=True,
    atomic=True,
    train_batch_size=32,
    eval_batch_size=32,
    max_epochs=1,
    pretrained_model="facebook/bart-large",
)
model = COMETBART(config)
train_graph = KnowledgeGraph.from_csv("data/atomic2020_data-feb2021/train.tsv", header=None, sep="\t")
val_graph = KnowledgeGraph.from_csv("data/atomic2020_data-feb2021/dev.tsv", header=None, sep="\t")
test_graph = KnowledgeGraph.from_csv("data/atomic2020_data-feb2021/test.tsv", header=None, sep="\t")
model.train(train_graph=train_graph, val_graph=val_graph, test_graph=test_graph, logger_name="wandb")
