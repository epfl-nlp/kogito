from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020.json")

model = COMETBART.from_pretrained("models/comet-bart/best_tfmr")
# model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
scores = model.evaluate(input_graph, batch_size=256, decode_method="greedy", num_generate=1)

print(scores)
