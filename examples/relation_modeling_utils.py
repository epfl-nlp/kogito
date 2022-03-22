import pandas as pd
from collections import defaultdict
from kogito.core.relation import PHYSICAL_RELATIONS, SOCIAL_RELATIONS, EVENT_RELATIONS

PHYSICAL_REL_LABEL = 0
EVENT_REL_LABEL = 1
SOCIAL_REL_LABEL = 2

def load_data(datapath):
    data = []
    head_label_map = defaultdict(set)

    with open(datapath) as f:
        for line in f:
            try:
                head, relation, _ = line.split('\t')

                label = PHYSICAL_REL_LABEL

                if relation in EVENT_RELATIONS:
                    label = EVENT_REL_LABEL
                elif relation in SOCIAL_RELATIONS:
                    label = SOCIAL_REL_LABEL

                head_label_map[head].add(label)
            except:
                pass
    
    for head, labels in head_label_map.items():
        # final_label = list(labels)[0] if len(labels) == 1 else 2 + sum(labels)
        final_label = max(labels)

        if len(labels) > 1:
            if EVENT_REL_LABEL in labels:
                final_label = EVENT_REL_LABEL

        data.append((head, final_label))

    return pd.DataFrame(data, columns=['text', 'label'])