from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet import COMETGPT2
import os

data_dir = os.environ.get("KOGITO_DATA_DIR")
model = COMETGPT2("gpt2-xl")
train_graph = KnowledgeGraph.from_csv(f"{data_dir}/atomic2020_data-feb2021/train.tsv", header=None, sep="\t")
val_graph = KnowledgeGraph.from_csv(f"{data_dir}/atomic2020_data-feb2021/dev.tsv", header=None, sep="\t")
model.train(
    train_graph=train_graph,
    val_graph=val_graph,
    batch_size=4,
    output_dir="/scratch/mete/models/comet-gpt2",
    log_wandb=True,
    epochs=3
)