import pandas as pd
from kogito.core.relation import PHYSICAL_RELATIONS, SOCIAL_RELATIONS, EVENT_RELATIONS

def load_data(datapath):
    data = []
    head_label_set = set()

    with open(datapath) as f:
        for line in f:
            try:
                head, relation, _ = line.split('\t')

                label = 0 

                if relation in EVENT_RELATIONS:
                    label = 1
                elif relation in SOCIAL_RELATIONS:
                    label = 2

                if (head, label) not in head_label_set:
                    data.append((head, label))
                    head_label_set.add((head, label))
            except:
                pass

    return pd.DataFrame(data, columns=['text', 'label'])