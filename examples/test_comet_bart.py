from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet_bart import COMETBART

input_graph = KnowledgeGraph.from_jsonl("./test_atomic2020.jsonl")

model = COMETBART.from_pretrained("./bart/best_tfmr")
output_graph = model.generate(input_graph)
output_graph.to_jsonl("test_atomic2020_res_cometbart.jsonl")