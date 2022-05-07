from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART
import json

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020.json")

model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
results = model.evaluate(input_graph, batch_size=128)

with open("score.json", "w") as f:
    f.write(json.dumps(results[0]))

with open("scores.json", "w") as f:
    f.write(json.dumps(results[1]))