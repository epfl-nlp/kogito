from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020_sample.json")

model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
scores = model.evaluate(input_graph, metrics=["bleu"], batch_size=256)

print(scores)
