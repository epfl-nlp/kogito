import csv
from kogito.core.knowledge import Knowledge, KnowledgeGraph
from kogito.models.gpt2.comet_gpt2 import COMETGPT2


def kg_graph_from_tsv(filepath):
    with open(filepath) as file:
        reader = csv.DictReader(file, delimiter='\t', fieldnames=['head', 'relation', 'tail', 'id1', 'id2', 'score'])
        graph = []
        for row in reader:
            graph.append(Knowledge(head=row['head'], relation=row['relation'], tails=[row['tail']]))
        
        return KnowledgeGraph(graph)

model = COMETGPT2()
train_graph = kg_graph_from_tsv("data/atomic2020/sample_train.tsv")
val_graph = kg_graph_from_tsv("data/atomic2020/sample_dev.tsv")
model.train(train_graph=train_graph, val_graph=val_graph, batch_size=2, output_dir='models')