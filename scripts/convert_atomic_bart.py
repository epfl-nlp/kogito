import sys
import csv

with open(sys.argv[1]) as file:
    reader = csv.DictReader(file, delimiter='\t', fieldnames=['head', 'relation', 'tail', 'id1', 'id2', 'score'])
    sources = []
    targets = []
    for row in reader:
        if len(row['tail']) > 0:
            sources.append(f"{row['head']} {row['relation']} [GEN]")
            targets.append(row['tail'])
    with open(sys.argv[2], 'w') as f:
        f.writelines('\n'.join(sources))
    with open(sys.argv[3], 'w') as f:
        f.writelines('\n'.join(targets))
