import torch
from collections import defaultdict
from transformers import BertTokenizer, BertModel

from kogito.evaluation.bert_score.utils import (
    get_idf_dict,
    bert_cos_score_idf,
    bert_types,
)


def score(
    cands,
    refs,
    bert="bert-base-multilingual-cased",
    num_layers=8,
    no_idf=False,
    batch_size=64,
):
    """
    BERTScore metric.
    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str): reference sentences
        - :param: `bert` (str): bert specification
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `no_idf` (bool): do not use idf weighting
        - :param: `batch_size` (int): bert score processing batch size
    """
    assert len(cands) == len(refs)
    assert bert in bert_types

    tokenizer = BertTokenizer.from_pretrained(bert)
    model = BertModel.from_pretrained(bert)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # drop unused layers
    model.encoder.layer = torch.nn.ModuleList(
        [layer for layer in model.encoder.layer[:num_layers]]
    )

    if no_idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[101] = 0
        idf_dict[102] = 0
    else:
        idf_dict = get_idf_dict(refs, tokenizer)

    all_preds = bert_cos_score_idf(
        model, refs, cands, tokenizer, idf_dict, device=device, batch_size=batch_size
    )

    P = all_preds[:, 0].cpu()
    R = all_preds[:, 1].cpu()
    F1 = all_preds[:, 2].cpu()

    return P, R, F1
