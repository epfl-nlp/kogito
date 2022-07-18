from kogito.models.bart.comet import COMETBART
from kogito.models.gpt2.comet import COMETGPT2
from kogito.models.gpt2.zeroshot import GPT2Zeroshot
from kogito.inference import CommonsenseInference
from kogito.core.relation import KnowledgeRelation
from kogito.core.processors.relation import SWEMRelationMatcher, DistilBERTRelationMatcher, BERTRelationMatcher

MODEL_MAP = {
    "comet-bart": COMETBART.from_pretrained("mismayil/comet-bart-ai2"),
    # "comet-gpt2": COMETGPT2.from_pretrained("mismayil/comet-gpt2-ai2"),
    # "gpt2": GPT2Zeroshot()
}

PROCESSOR_MAP = {
    "swem_relation_matcher": SWEMRelationMatcher("swem_relation_matcher"),
    "distilbert_relation_matcher": DistilBERTRelationMatcher("distilbert_relation_matcher"),
    "bert_relation_matcher": BERTRelationMatcher("bert_relation_matcher"),
}

def infer(data):
    text = data.get("text")
    model = MODEL_MAP.get(data.get("model"))
    heads = data.get("heads")
    relations = data.get("relations")
    extract_heads = data.get("extractHeads", True)
    match_relations = data.get("matchRelations", True)
    dry_run = data.get("dryRun", False)
    head_procs = data.get("headProcs", [])
    rel_procs = data.get("relProcs", [])

    csi = CommonsenseInference(language="en_core_web_sm")
    csi_head_procs = csi.processors["head"]
    csi_rel_procs = csi.processors["relation"]

    for proc in set(csi_head_procs).difference(set(head_procs)):
        csi.remove_processor(proc)

    for proc in set(csi_rel_procs).difference(set(rel_procs)):
        csi.remove_processor(proc)
 
    for proc in set(head_procs).difference(set(csi_head_procs)):
        csi.add_processor(PROCESSOR_MAP[proc])

    for proc in set(rel_procs).difference(set(csi_rel_procs)):
        csi.add_processor(PROCESSOR_MAP[proc])

    if relations:
        for i in range(len(relations)):
            relations[i] = KnowledgeRelation.from_text(relations[i])

    output_graph = csi.infer(text=text,
                             model=model,
                             heads=heads,
                             relations=relations,
                             extract_heads=extract_heads,
                             match_relations=match_relations,
                             dry_run=dry_run)

    return [kg.to_json() for kg in output_graph]