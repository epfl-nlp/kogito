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
    max_epochs=5,
    pretrained_model="facebook/bart-large",
)
model = COMETBART(config)
train_graph = KnowledgeGraph.from_jsonl("data/persona/persona_atomic_train.jsonl", tails_attr="tail")
val_graph = KnowledgeGraph.from_jsonl("data/persona/persona_atomic_val.jsonl", tails_attr="tail")
test_graph = KnowledgeGraph.from_jsonl("data/persona/persona_atomic_test.jsonl", tails_attr="tail")
model.train(train_graph=train_graph, val_graph=val_graph, test_graph=test_graph, logger_name="wandb")
