from typing import List, Tuple

from kogito.evaluation.bleu.bleu import Bleu
from kogito.evaluation.meteor.meteor import Meteor
from kogito.evaluation.rouge.rouge import Rouge
from kogito.evaluation.cider.cider import Cider
from kogito.evaluation.bert_score.bert_score import BertScore
from kogito.core.knowledge import Knowledge


METRIC_MAP = {
    "bleu": (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    "meteor": (Meteor(), "METEOR"),
    "rouge": (Rouge(), "ROUGE_L"),
    "cider": (Cider(), "CIDEr"),
    "bert-score": (BertScore(), "Bert Score"),
}


class Evaluator:
    def __init__(self, gts, res, metrics):
        self.gts = gts
        self.res = res
        self.scorers = [METRIC_MAP[metric] for metric in metrics]

    def evaluate(self):
        score_dict = {}

        for scorer, method in self.scorers:
            score, _ = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    score_dict[m] = str(sc)
            else:
                score_dict[method] = score

        return score_dict


def topk_eval(data: List[Tuple[Knowledge, List[str]]], metrics, k=1):
    topk_gts = {}
    topk_res = {}

    for i, (kg, reference) in enumerate(data):
        for j, g in enumerate(kg.tails[:k]):
            key = str(i) + "_" + str(j)
            topk_gts[key] = reference
            topk_res[key] = [g]

    evaluator = Evaluator(topk_gts, topk_res, metrics)
    scores = evaluator.evaluate()

    return scores
