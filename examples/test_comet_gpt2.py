from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet import COMETGPT2

model = COMETGPT2.from_pretrained("mismayil/comet-gpt2-ai2")
input_graph = KnowledgeGraph.from_jsonl("./test_atomic2020_sample.json")
output_graph = model.generate(input_graph)
output_graph.to_jsonl("results/test_atomic2020_res_cometgpt2_sample2.json")
