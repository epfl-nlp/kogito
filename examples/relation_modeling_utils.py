import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
import spacy
from torch.nn.utils.rnn import pad_sequence

from kogito.core.relation import PHYSICAL_RELATIONS, SOCIAL_RELATIONS, EVENT_RELATIONS

PHYSICAL_REL_LABEL = 0
EVENT_REL_LABEL = 1
SOCIAL_REL_LABEL = 2

def load_data(datapath, multi_label=False):
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
        if multi_label:
            final_label = [1 if label in labels else 0 for label in range(3)]
            data.append((head, final_label))
        else:
            final_label = max(labels)

            if len(labels) > 1:
                if EVENT_REL_LABEL in labels:
                    final_label = EVENT_REL_LABEL

            data.append((head, final_label))

    return pd.DataFrame(data, columns=['text', 'label'])


def create_emb_matrix(embedding_dim=100):
    glove = pd.read_csv(f'data/glove/glove.6B.{embedding_dim}d.txt', sep=" ", quoting=3, header=None, index_col=0)
    vocab = {'<pad>': 0, '<unk>': 1}
    embeddings = np.zeros((len(glove) + 2, embedding_dim))
    embeddings[0] = np.zeros(embedding_dim)
    embeddings[1] = np.zeros(embedding_dim)

    for index, (key, val) in tqdm(enumerate(glove.T.items()), total=len(glove)):
        vocab[key] = index + 2
        embeddings[index+2] = val.to_numpy()

    return vocab, embeddings


class HeadDataset(Dataset):
    def __init__(self, df, vocab, embedding_matrix=None, apply_pooling=False, pooling="avg"):
        nlp = spacy.load("en_core_web_sm")
        self.texts = []

        if apply_pooling:
            # Apply pooling directly without padding
            self.labels = []
            self.features = []

            for index, text in enumerate(df['text']):
                embedding = text_to_embedding(text, vocab=vocab, embedding_matrix=embedding_matrix, nlp=nlp)
                if embedding is not None:
                    self.features.append(embedding)
                    self.labels.append(df['label'][index])
                    self.texts.append(text)
            
            self.labels = np.asarray(self.labels)
        else:
            # Pad sequences
            self.texts = df['text']
            self.labels = np.asarray(df['label'].to_list())
            self.features = pad_sequence([torch.tensor([vocab.get(token.text, 1) for token in nlp(text)], dtype=torch.int) for text in df['text']],
                                    batch_first=True)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MaxPool(nn.Module):
    def forward(self, X):
        values, _ = torch.max(X, dim=1)
        return values


class AvgPool(nn.Module):
    def forward(self, X):
        return torch.mean(X, dim=1)


def text_to_embedding(text, vocab, embedding_matrix, pooling="max", nlp=None):
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    vectors = []
    for token in doc:
        if token.text in vocab:
            vectors.append(embedding_matrix[vocab[token.text]])
    
    if vectors:
        if pooling == "max":
            return np.amax(np.array(vectors, dtype=np.float32), axis=0)
        return np.mean(vectors, axis=0, dtype=np.float32)