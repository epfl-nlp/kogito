from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020.json")

model = COMETBART.from_pretrained("models/comet-bart/best_tfmr")
output_graph = model.generate(input_graph)
output_graph.to_jsonl("cometbart_results_test_atomic2020.json")
