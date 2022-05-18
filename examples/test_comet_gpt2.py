from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet import COMETGPT2

model = COMETGPT2.from_pretrained("/scratch/mete/models/comet-gpt2/final_model")
input_graph = KnowledgeGraph.from_jsonl("./test_atomic2020.json")
output_graph = model.generate(input_graph, batch_size=8)
output_graph.to_jsonl("test_atomic2020_res_cometgpt2.json")
