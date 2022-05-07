#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help


from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


class Meteor:
    def __init__(self):
        pass

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        imgIds = gts.keys()
        scores = []

        for i in imgIds:
            assert len(res[i]) == 1
            score = round(
                meteor_score(
                    [word_tokenize(s) for s in gts[i]], word_tokenize(res[i][0])
                ),
                4,
            )
            scores.append(score)

        return sum(scores) / len(scores), scores

    def method(self):
        return "METEOR"
