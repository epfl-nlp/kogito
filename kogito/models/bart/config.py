from typing import Optional
from dataclasses import dataclass, asdict

BART_TASKS = ["summarization", "translation"]
FP16_OPT_LEVELS = ["O0", "O1", "O2", "O3"]


@dataclass
class COMETBARTConfig:
    output_dir: str = None
    fp16: bool = False
    fp16_opt_level: str = "O2"
    tpu_cores: Optional[int] = None
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    seed: int = 42
    max_source_length: int = 48
    max_target_length: int = 24
    val_max_target_length: int = 24
    test_max_target_length: int = 24
    freeze_encoder: bool = False
    freeze_embeds: bool = False
    sortish_sampler: bool = True
    n_train: int = -1
    n_val: int = 500
    n_test: int = -1
    task: str = "summarization"
    src_lang: str = ""
    tgt_lang: str = ""
    atomic: bool = True
    pretrained_model: str = "facebook/bart-large"
    pretrained_config: str = None
    pretrained_tokenizer: str = None
    cache_dir: str = ""
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    num_workers: int = 2
    max_epochs: int = 3
    train_batch_size: int = 32
    eval_batch_size: int = 32
    gpus: int = 1
    decoder_start_token_id: Optional[int] = None
    num_train_epochs: int = 3

    def __dict__(self):
        return asdict(self)
