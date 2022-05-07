from typing import List, Tuple

from kogito.evaluation.bleu.bleu import Bleu
from kogito.evaluation.meteor.meteor import Meteor
from kogito.evaluation.rouge.rouge import Rouge
from kogito.evaluation.cider.cider import Cider
from kogito.evaluation.bert_score.bert_score import BertScore
from kogito.core.knowledge import Knowledge


class Evaluator:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (BertScore(), "Bert Score")
        ]

        score_dict = {}
        scores_dict = {}

        for scorer, method in scorers:
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    score_dict[m] = str(sc)
                    scores_dict[m] = list(scs)
            else:
                score_dict[method] = score
                scores_dict[method] = list(scores)

        return score_dict, scores_dict


def topk_eval(data: List[Tuple[Knowledge, List[str]]], k=1):
    topk_gts = {}
    topk_res = {}

    for i, kg, reference in enumerate(data):
        for j, g in enumerate(kg.tails[:k]):
            key = str(i) + "_" + str(j)
            topk_gts[key] = reference
            topk_res[key] = [g]

    evaluator = Evaluator(topk_gts, topk_res)
    score, scores = evaluator.evaluate()
    
    return score, scores
