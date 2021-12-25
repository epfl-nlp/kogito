from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.zeroshot import GPT2Zeroshot

input_graph = KnowledgeGraph.from_jsonl("./test_atomic2020.jsonl")

model = GPT2Zeroshot()
output_graph = model.generate(input_graph)
output_graph.to_jsonl("test_atomic2020_res_zeroshot.jsonl")
