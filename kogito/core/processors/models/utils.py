import torchmetrics
import numpy as np
import spacy


def text_to_embedding(text, vocab, embedding_matrix, pooling="max", lang=None):
    if not lang:
        lang = spacy.load("en_core_web_sm")

    doc = lang(text)
    vectors = []
    for token in doc:
        if token.text in vocab:
            vectors.append(embedding_matrix[vocab[token.text]])

    if vectors:
        if pooling == "max":
            return np.amax(np.array(vectors, dtype=np.float32), axis=0)
        return np.mean(vectors, axis=0, dtype=np.float32)


class Evaluator:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.metrics = dict(
            train_accuracy=torchmetrics.Accuracy(),
            # (weighted)
            train_precision=torchmetrics.Precision(num_classes=3, average="weighted"),
            train_recall=torchmetrics.Recall(num_classes=3, average="weighted"),
            train_f1=torchmetrics.F1Score(num_classes=3, average="weighted"),
            # (micro)
            train_precision_micro=torchmetrics.Precision(
                num_classes=3, average="micro"
            ),
            train_recall_micro=torchmetrics.Recall(num_classes=3, average="micro"),
            train_f1_micro=torchmetrics.F1Score(num_classes=3, average="micro"),
            # (macro)
            train_precision_macro=torchmetrics.Precision(
                num_classes=3, average="macro"
            ),
            train_recall_macro=torchmetrics.Recall(num_classes=3, average="macro"),
            train_f1_macro=torchmetrics.F1Score(num_classes=3, average="macro"),
            # (per class)
            train_precision_class=torchmetrics.Precision(num_classes=3, average="none"),
            train_recall_class=torchmetrics.Recall(num_classes=3, average="none"),
            train_f1_class=torchmetrics.F1Score(num_classes=3, average="none"),
            # Validation metrics
            val_accuracy=torchmetrics.Accuracy(),
            # (weighted)
            val_precision=torchmetrics.Precision(num_classes=3, average="weighted"),
            val_recall=torchmetrics.Recall(num_classes=3, average="weighted"),
            val_f1=torchmetrics.F1Score(num_classes=3, average="weighted"),
            # (micro)
            val_precision_micro=torchmetrics.Precision(num_classes=3, average="micro"),
            val_recall_micro=torchmetrics.Recall(num_classes=3, average="micro"),
            val_f1_micro=torchmetrics.F1Score(num_classes=3, average="micro"),
            # (macro)
            val_precision_macro=torchmetrics.Precision(num_classes=3, average="macro"),
            val_recall_macro=torchmetrics.Recall(num_classes=3, average="macro"),
            val_f1_macro=torchmetrics.F1Score(num_classes=3, average="macro"),
            # (per class)
            val_precision_class=torchmetrics.Precision(num_classes=3, average="none"),
            val_recall_class=torchmetrics.Recall(num_classes=3, average="none"),
            val_f1_class=torchmetrics.F1Score(num_classes=3, average="none"),
            # Test metrics
            test_accuracy=torchmetrics.Accuracy(),
            # (weighted)
            test_precision=torchmetrics.Precision(num_classes=3, average="weighted"),
            test_recall=torchmetrics.Recall(num_classes=3, average="weighted"),
            test_f1=torchmetrics.F1Score(num_classes=3, average="weighted"),
            # (micro)
            test_precision_micro=torchmetrics.Precision(num_classes=3, average="micro"),
            test_recall_micro=torchmetrics.Recall(num_classes=3, average="micro"),
            test_f1_micro=torchmetrics.F1Score(num_classes=3, average="micro"),
            # (macro)
            test_precision_macro=torchmetrics.Precision(num_classes=3, average="macro"),
            test_recall_macro=torchmetrics.Recall(num_classes=3, average="macro"),
            test_f1_macro=torchmetrics.F1Score(num_classes=3, average="macro"),
            # (per class)
            test_precision_class=torchmetrics.Precision(num_classes=3, average="none"),
            test_recall_class=torchmetrics.Recall(num_classes=3, average="none"),
            test_f1_class=torchmetrics.F1Score(num_classes=3, average="none"),
        )

    def log_metrics(self, preds, y, type):
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(type):
                metric(preds.cpu(), y.cpu())
                value = metric.compute()
                if len(value.shape) > 0:
                    for idx, val in enumerate(value):
                        self.log(
                            f"{metric_name}_{idx}", val, on_epoch=True, on_step=False
                        )
                else:
                    self.log(metric_name, value, on_epoch=True, on_step=False)
