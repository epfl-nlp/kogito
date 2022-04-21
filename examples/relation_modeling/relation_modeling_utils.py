import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import ast

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
import spacy
from torch.nn.utils.rnn import pad_sequence
import torchmetrics

from kogito.core.relation import PHYSICAL_RELATIONS, SOCIAL_RELATIONS, EVENT_RELATIONS, KnowledgeRelation
from spacy.lang.en.stop_words import STOP_WORDS

IGNORE_WORDS = ["PersonX", "PersonY", "PersonZ", "_", "'", "-"]

PHYSICAL_REL_LABEL = 0
EVENT_REL_LABEL = 1
SOCIAL_REL_LABEL = 2

GLOBAL_DOCS = {}

def load_data(datapath, multi_label=False):
    data = []
    head_label_map = defaultdict(set)

    with open(datapath) as f:
        for line in f:
            try:
                head, rel, _ = line.split('\t')
                relation = KnowledgeRelation.from_text(rel)
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


class SWEMHeadDataset(Dataset):
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


def get_timestamp():
    now = datetime.now()
    return now.strftime('%Y%m%dH%H%M')


def explode_labels(df):
    df['label_0'] = df.label.apply(lambda l: l[0])
    df['label_1'] = df.label.apply(lambda l: l[1])
    df['label_2'] = df.label.apply(lambda l: l[2])
    return df


def get_class_dist_report(df):
    class_0 = df.label.apply(lambda l: l[0])
    class_1 = df.label.apply(lambda l: l[1])
    class_2 = df.label.apply(lambda l: l[2])
    report = {}

    for class_x_idx, class_x in enumerate([class_0, class_1, class_2]):
        for class_y_idx, class_y in enumerate([class_0, class_1, class_2]):
            for label_x in [0, 1]:
                for label_y in [0, 1]:
                    sub_df = df[(class_x == label_x)]
                    subsub_df = df[(class_x == label_x) & (class_y == label_y)]
                    report[(f"class_{class_x_idx}", label_x)] = len(sub_df) / len(df)
                    report[(f"class_{class_x_idx}", f"class_{class_y_idx}", label_x, label_y)] = len(subsub_df) / len(df)
    
    return report

class HeuristicClassifier:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", exclude=["ner"])

    def predict(self, data):
        preds = []

        for text in tqdm(data.text):
            doc = self.nlp(text.replace("_", "").replace("  ", " "))
            is_sentence = False

            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    is_sentence = True
                    break
            
            if is_sentence:
                preds.append([0, 1, 1])
            else:
                preds.append([1, 0, 0])
        
        return preds

def report_metrics(preds, y):
    accuracy = torchmetrics.Accuracy()
    precision = torchmetrics.Precision(num_classes=3, average="weighted")
    recall = torchmetrics.Recall(num_classes=3, average="weighted")
    f1 = torchmetrics.F1Score(num_classes=3, average="weighted")
    print(f'Accuracy={accuracy(preds, y).item():.3f}, precision={precision(preds, y).item():.3f}, recall={recall(preds, y).item():.3f}, f1={f1(preds, y).item():.3f}')

def create_vocab(data, include_stopwords=True):
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    vocab = defaultdict(int)

    for text in tqdm(data.text, total=len(data)):
        doc = get_doc(nlp, text)
        for token in doc:
            if token.text not in IGNORE_WORDS and (include_stopwords or token.text not in STOP_WORDS):
                vocab[token.lemma_] += 1
    
    return vocab

def get_doc(nlp, text):
    doc = GLOBAL_DOCS.get(text)
    if doc:
        return doc
    doc = nlp(text)
    GLOBAL_DOCS[text] = doc
    return doc

def load_fdata(datapath):
    data = pd.read_csv(datapath)
    data['label'] = data['label'].apply(ast.literal_eval)
    return data


class Evaluator:
    def __init__(self) -> None:
        super().__init__()
        self.metrics = dict(
            train_accuracy = torchmetrics.Accuracy(),
            # (weighted)
            train_precision = torchmetrics.Precision(num_classes=3, average='weighted'),
            train_recall = torchmetrics.Recall(num_classes=3, average='weighted'),
            train_f1 = torchmetrics.F1Score(num_classes=3, average='weighted'),
            # (micro)
            train_precision_micro = torchmetrics.Precision(num_classes=3, average='micro'),
            train_recall_micro = torchmetrics.Recall(num_classes=3, average='micro'),
            train_f1_micro = torchmetrics.F1Score(num_classes=3, average='micro'),
            # (macro)
            train_precision_macro = torchmetrics.Precision(num_classes=3, average='macro'),
            train_recall_macro = torchmetrics.Recall(num_classes=3, average='macro'),
            train_f1_macro = torchmetrics.F1Score(num_classes=3, average='macro'),
            # (per class)
            train_precision_class = torchmetrics.Precision(num_classes=3, average='none'),
            train_recall_class = torchmetrics.Recall(num_classes=3, average='none'),
            train_f1_class = torchmetrics.F1Score(num_classes=3, average='none'),

            # Validation metrics
            val_accuracy = torchmetrics.Accuracy(),
            # (weighted)
            val_precision = torchmetrics.Precision(num_classes=3, average='weighted'),
            val_recall = torchmetrics.Recall(num_classes=3, average='weighted'),
            val_f1 = torchmetrics.F1Score(num_classes=3, average='weighted'),
            # (micro)
            val_precision_micro = torchmetrics.Precision(num_classes=3, average='micro'),
            val_recall_micro = torchmetrics.Recall(num_classes=3, average='micro'),
            val_f1_micro = torchmetrics.F1Score(num_classes=3, average='micro'),
            # (macro)
            val_precision_macro = torchmetrics.Precision(num_classes=3, average='macro'),
            val_recall_macro = torchmetrics.Recall(num_classes=3, average='macro'),
            val_f1_macro = torchmetrics.F1Score(num_classes=3, average='macro'),
            # (per class)
            val_precision_class = torchmetrics.Precision(num_classes=3, average='none'),
            val_recall_class = torchmetrics.Recall(num_classes=3, average='none'),
            val_f1_class = torchmetrics.F1Score(num_classes=3, average='none'),

            # Test metrics
            test_accuracy = torchmetrics.Accuracy(),
            # (weighted)
            test_precision = torchmetrics.Precision(num_classes=3, average='weighted'),
            test_recall = torchmetrics.Recall(num_classes=3, average='weighted'),
            test_f1 = torchmetrics.F1Score(num_classes=3, average='weighted'),
            # (micro)
            test_precision_micro = torchmetrics.Precision(num_classes=3, average='micro'),
            test_recall_micro = torchmetrics.Recall(num_classes=3, average='micro'),
            test_f1_micro = torchmetrics.F1Score(num_classes=3, average='micro'),
            # (macro)
            test_precision_macro = torchmetrics.Precision(num_classes=3, average='macro'),
            test_recall_macro = torchmetrics.Recall(num_classes=3, average='macro'),
            test_f1_macro = torchmetrics.F1Score(num_classes=3, average='macro'),
            # (per class)
            test_precision_class = torchmetrics.Precision(num_classes=3, average='none'),
            test_recall_class = torchmetrics.Recall(num_classes=3, average='none'),
            test_f1_class = torchmetrics.F1Score(num_classes=3, average='none')
        )
    
    def log_metrics(self, preds, y, type):
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(type):
                metric(preds.cpu(), y.cpu())
                self.log(metric_name, metric.compute(), on_epoch=True, on_step=False)