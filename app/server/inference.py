from kogito.models.bart.comet import COMETBART
from kogito.models.gpt2.comet import COMETGPT2
from kogito.models.gpt2.zeroshot import GPT2Zeroshot
from kogito.inference import CommonsenseInference

MODEL_MAP = {
    "comet-bart": COMETBART.from_pretrained("mismayil/comet-bart-ai2"),
    "comet-gpt2": COMETGPT2.from_pretrained("mismayil/comet-bart-gpt2"),
    "gpt2": GPT2Zeroshot()
}

def infer(data):
    text = data.get("text")
    model = MODEL_MAP.get(data.get("model"))
    csi = CommonsenseInference(language="en_core_web_sm")
    output_graph = csi.infer(text=text, model=model)
    return [kg.to_json() for kg in output_graph]