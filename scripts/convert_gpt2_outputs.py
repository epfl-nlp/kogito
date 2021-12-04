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
    predictions = []
    for line in file:
        pred = json.loads(line)
        predictions.append({
            'head': pred['fact']['head'],
            'relation': pred['fact']['relation'],
            'tails': pred['fact']['tails'],
            'generations': pred['generations'],
            'greedy': pred['generations'][0]
        })
    write_jsonl(sys.argv[2], predictions)