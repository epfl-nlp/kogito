import glob
import os
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm
import pytorch_lightning as pl
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from kogito.core.knowledge import KnowledgeGraph
from kogito.core.relation import KG_RELATIONS
from kogito.core.utils import (
    pickle_save,
    chunks,
    trim_batch,
)
from kogito.core.callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback
from kogito.core.model import KnowledgeModel
from kogito.models.bart.config import COMETBARTConfig
from kogito.models.bart.utils import (
    generic_train,
    SummarizationModule,
    TranslationModule,
    use_task_specific_params,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class COMETBART(KnowledgeModel):
    """COMET knowledge model based on BART"""

    def __init__(self, config: COMETBARTConfig, **kwargs) -> None:
        """Initialize COMET model

        Args:
            config (COMETBARTConfig): Config to use
        """
        self.model = None
        self.tokenizer = None
        self.config = config
        self.kwargs = kwargs

    def train(
        self,
        train_graph: KnowledgeGraph,
        val_graph: KnowledgeGraph,
        test_graph: KnowledgeGraph,
        logger_name: str = "default",
    ) -> KnowledgeModel:
        """Train a COMET model

        Args:
            train_graph (KnowledgeGraph): Training dataset
            val_graph (KnowledgeGraph): Validation dataset
            test_graph (KnowledgeGraph): Test dataset
            logger_name (str, optional): Logger name to use. Accepted values: ["wandb", "default"]
                                         Defaults to "default".

        Raises:
            ValueError: When config.task is not recognized

        Returns:
            KnowledgeModel: Trained knowledge model
        """
        Path(self.config.output_dir).mkdir(exist_ok=True)

        if self.config.task == "summarization":
            self.model: SummarizationModule = SummarizationModule(
                self.config, train_graph, val_graph, test_graph, **self.kwargs
            )
        elif self.config.task == "translation":
            self.model: SummarizationModule = TranslationModule(
                self.config, train_graph, val_graph, test_graph, **self.kwargs
            )
        else:
            raise ValueError("Unrecognized task")

        if self.config.atomic:
            self.model.tokenizer.add_tokens(
                [str(relation) for relation in KG_RELATIONS]
            )
            self.model.model.resize_token_embeddings(len(self.model.tokenizer))

        if (
            logger_name == "default"
            or str(self.config.output_dir).startswith("/tmp")
            or str(self.config.output_dir).startswith("/var")
        ):
            logger = True  # don't pollute wandb logs unnecessarily

        elif logger_name == "wandb":
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(
                name=self.model.output_dir.name, project=self.model.output_dir.name
            )

        elif logger_name == "wandb_shared":
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(
                name=self.model.output_dir.name,
                project=f"hf_{self.model.output_dir.name}",
            )

        trainer: pl.Trainer = generic_train(
            self.model,
            self.config,
            logging_callback=Seq2SeqLoggingCallback(),
            checkpoint_callback=get_checkpoint_callback(
                self.config.output_dir, self.model.val_metric
            ),
            logger=logger,
        )
        pickle_save(self.model.config, self.model.output_dir / "config.pkl")

        self.model.config.test_checkpoint = ""
        checkpoints = list(
            sorted(
                glob.glob(
                    os.path.join(self.config.output_dir, "*.ckpt"), recursive=True
                )
            )
        )
        if checkpoints:
            self.model.config.test_checkpoint = checkpoints[-1]
            trainer.resume_from_checkpoint = checkpoints[-1]
        trainer.logger.log_hyperparams(asdict(self.model.config))

        trainer.test(self.model)

    def generate(
        self,
        input_graph: KnowledgeGraph,
        decode_method: str = "beam",
        num_generate: int = 3,
        batch_size: int = 64,
    ) -> KnowledgeGraph:
        """Generate inferences from the model

        Args:
            input_graph (KnowledgeGraph): Input dataset
            decode_method (str, optional): Decoding method. Accepts ["beam", "greedy"]. Defaults to "beam".
            num_generate (int, optional): Number of inferences to generate. Defaults to 3.
            batch_size (int, optional): Batch size to use. Defaults to 64.

        Returns:
            KnowledgeGraph: Complete knowledge graph
        """
        with torch.no_grad():
            outputs = []
            for kg_batch in tqdm(list(chunks(input_graph, batch_size))):
                queries = []
                for kg_input in kg_batch:
                    queries.append(kg_input.to_query(decode_method=decode_method))
                batch = self.tokenizer(
                    queries, return_tensors="pt", truncation=True, padding="max_length"
                ).to(device)
                input_ids, attention_mask = trim_batch(
                    **batch, pad_token_id=self.tokenizer.pad_token_id
                )

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.config.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                )

                output = self.tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for kg_input, generations in zip(
                    kg_batch, list(chunks(output, num_generate))
                ):
                    output_kg = kg_input.copy()
                    output_kg.tails = generations
                    outputs.append(output_kg)

            return KnowledgeGraph(outputs)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, task: str = "summarization"
    ) -> KnowledgeModel:
        """Load pretrained model

        Args:
            model_name_or_path (str): HuggingFace model name or local model path
            task (str, optional): Task used in training. Defaults to "summarization".

        Returns:
            KnowledgeModel: Loaded knowledge model
        """
        config = COMETBARTConfig(task=task, decoder_start_token_id=None)
        comet_bart = cls(config)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        use_task_specific_params(model, task)
        comet_bart.model = model
        comet_bart.tokenizer = tokenizer
        model.to(device)
        return comet_bart

    def save_pretrained(self, save_path: str) -> None:
        """Save pretrained model

        Args:
            save_path (str): Directory path to save model to
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
