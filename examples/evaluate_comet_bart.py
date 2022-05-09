from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020.json")

model = COMETBART.from_pretrained("models/comet-bart/best_tfmr")
scores = model.evaluate(input_graph, top_k=10, batch_size=64, decode_method="beam", num_generate=10)

import json
with open("comet-bart-local-scores3.json", "w") as f:
    f.write(json.dumps(scores))

print(scores)
