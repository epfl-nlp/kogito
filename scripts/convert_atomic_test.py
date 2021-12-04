import sys
import json

def write_items(output_file, items):
    with open(output_file, 'w') as f:
        for concept in items:
            f.write(concept + "\n")
    f.close()


def write_jsonl(f, d):
    write_items(f, [json.dumps(r) for r in d])


with open(sys.argv[1]) as file:
    prefixes = []
    for line in file:
        parts = line.split('\t')
        head_relation = parts[0].split('@@')
        head = head_relation[0].strip()
        relation = head_relation[1].strip()
        tails = [t.strip() for t in parts[1].split('|')]
        prefixes.append({'relation': relation, 'head': head, 'tails': tails})
    write_jsonl(sys.argv[2], prefixes)