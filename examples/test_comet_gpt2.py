from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet_gpt2 import COMETGPT2

model = COMETGPT2.from_pretrained("./comet_gpt2_pretrained")
input_graph = KnowledgeGraph.from_jsonl("./test_atomic2020.jsonl")
output_graph = model.generate(input_graph)
output_graph.to_jsonl("test_atomic2020_res_cometgpt2.jsonl")
