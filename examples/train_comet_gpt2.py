from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet import COMETGPT2
import os

os.environ["WANDB_DISABLED"] = "true"
model = COMETGPT2("gpt2-xl")
train_graph = KnowledgeGraph.from_csv("/root/kogito/examples/data/atomic2020/sample_train.tsv", header=None, sep="\t")
val_graph = KnowledgeGraph.from_csv("/root/kogito/examples/data/atomic2020/sample_dev.tsv", header=None, sep="\t")
model.train(
    train_graph=train_graph, val_graph=val_graph, batch_size=1, output_dir="models/comet-gpt2"
)
