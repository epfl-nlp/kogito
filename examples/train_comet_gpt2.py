from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet import COMETGPT2

model = COMETGPT2("gpt2-xl")
train_graph = KnowledgeGraph.from_csv("data/atomic2020/sample_train.tsv", header=None, sep="\t")
val_graph = KnowledgeGraph.from_csv("data/atomic2020/sample_dev.tsv", header=None, sep="\t")
model.train(
    train_graph=train_graph, val_graph=val_graph, batch_size=32, output_dir="models/comet-gpt2", epochs=1, lr=5e-5
)
