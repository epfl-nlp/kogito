from typing import Optional
import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from kogito.models.gpt2.utils import GPT2Finetuner
from kogito.core.dataset import KnowledgeDataset
from kogito.core.model import KnowledgeModel
from kogito.core.knowledge import KnowledgeGraph, GEN_TOKEN, EOS_TOKEN, PAD_TOKEN
from kogito.core.relation import KG_RELATIONS

device = "cuda" if cuda.is_available() else "cpu"


class COMETGPT2(KnowledgeModel):
    """COMET model based on GPT-2"""

    def __init__(self, model_name_or_path: str = "gpt2") -> None:
        """Initialize COMET model

        Args:
            model_name_or_path (str, optional): HuggingFace model name or local model path. Defaults to "gpt2".
        """
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model.to(device)

    def train(
        self,
        train_graph: KnowledgeGraph,
        val_graph: KnowledgeGraph,
        batch_size: int = 8,
        in_len: int = 16,
        out_len: int = 34,
        summary_len: int = 0,
        epochs: int = 1,
        lr: float = 5e-5,
        seed: int = 42,
        log_wandb: bool = False,
        output_dir: Optional[str] = None,
    ) -> KnowledgeModel:
        """Train a COMET model

        Args:
            train_graph (KnowledgeGraph): Training dataset
            val_graph (KnowledgeGraph): Validation dataset
            batch_size (int, optional): Batch size. Defaults to 2.
            in_len (int, optional): Input length. Defaults to 16.
            out_len (int, optional): Output length. Defaults to 34.
            summary_len (int, optional): Summary length. Defaults to 0.
            epochs (int, optional): Number of epochs. Defaults to 3.
            lr (float, optional): Learning rate. Defaults to 1e-5.
            seed (int, optional): Random seed. Defaults to 42.
            log_wandb (bool, optional): Whether to log to wandb. Defaults to False.
            output_dir (Optional[str], optional): Directory to save intermediate model checkpoints. Defaults to None.

        Returns:
            KnowledgeModel: Trained knowledge model
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        self.tokenizer.add_special_tokens(
            {
                "eos_token": EOS_TOKEN,
                "pad_token": PAD_TOKEN,
                "additional_special_tokens": [
                    str(relation) for relation in KG_RELATIONS
                ]
                + [GEN_TOKEN],
            }
        )

        train_dataset = KnowledgeDataset(
            train_graph,
            tokenizer=self.tokenizer,
            source_len=out_len,
            target_len=summary_len,
        )
        val_dataset = KnowledgeDataset(
            val_graph,
            tokenizer=self.tokenizer,
            source_len=in_len,
            target_len=out_len - in_len,
            is_eval=True,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
        val_params = {"batch_size": batch_size, "shuffle": False, "num_workers": 0}

        train_loader = DataLoader(train_dataset, **train_params, drop_last=True)
        val_loader = DataLoader(val_dataset, **val_params, drop_last=True)

        logger = True

        if log_wandb:
            config = {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "seed": seed,
                "in_len": in_len,
                "summary_len": summary_len,
                "out_len": out_len,
            }
            logger = WandbLogger(project="kogito-comet-gpt2")
            logger.experiment.config.update(config)

        finetuner = GPT2Finetuner(model=self.model, learning_rate=lr)
        trainer = pl.Trainer(
            default_root_dir=output_dir,
            max_epochs=epochs,
            logger=logger,
            accelerator="auto",
        )
        trainer.fit(
            finetuner, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        self.save_pretrained(f"{output_dir}/final_model")

        return self.model

    def generate(
        self,
        input_graph: KnowledgeGraph,
        max_length: int = 34,
        in_len: int = 16,
        out_len: int = 34,
        top_k: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_beams: int = 10,
        num_return_sequences: int = 10,
    ) -> KnowledgeGraph:
        """Generate inferences from knowledge model

        Args:
            input_graph (KnowledgeGraph): Input dataset
            max_length (int, optional): Maximum output length. Defaults to 34.
            in_len (int, optional): Input length. Defaults to 16.
            out_len (int, optional): Output length. Defaults to 34.
            top_k (int, optional): Top k inferences to consider. Defaults to 1.
            temperature (float, optional): GPT-2 temperature parameter. Defaults to 1.0.
            top_p (float, optional): GPT-2 top_p parameter. Defaults to 0.9.
            repetition_penalty (float, optional): GPT-2 repetition_penalty parameter. Defaults to 1.0.
            num_beams (int, optional): GPT-2 num_beams parameter. Defaults to 10.
            num_return_sequences (int, optional): GPT-2 num_return_sequences parameter. Defaults to 10.

        Returns:
            KnowledgeGraph: Completed knowledge graph
        """
        params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
        dataset = KnowledgeDataset(
            input_graph,
            tokenizer=self.tokenizer,
            source_len=in_len,
            target_len=out_len - in_len,
            is_eval=True,
        )
        loader = DataLoader(dataset, **params, drop_last=False)

        self.model.eval()

        outputs = []

        with torch.no_grad():
            for input_kg, data in zip(input_graph, loader):
                ids = data["source_ids"].to(device)
                mask = data["source_mask"].to(device)

                generated_ids = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    temperature=temperature,
                    do_sample=False,
                    max_length=max_length,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences if top_k > 1 else 1,
                    num_beams=num_beams,
                )

                generations = [
                    self.tokenizer.decode(g, clean_up_tokenization_spaces=True)
                    for g in generated_ids
                ]

                generations = [g[g.find(GEN_TOKEN)+len(GEN_TOKEN):g.find(EOS_TOKEN)].strip() for g in generations]

                output_kg = input_kg.copy()
                output_kg.tails = generations
                outputs.append(output_kg)

        return KnowledgeGraph(outputs)

    def save_pretrained(self, save_path: str) -> None:
        """Save pretrained model

        Args:
            save_path (str): Directory to save model to
        """
        if save_path:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> KnowledgeModel:
        """Load pretrained model

        Args:
            model_name_or_path (str): HuggingFace model name or local model path

        Returns:
            KnowledgeModel: Loaded knowledge model
        """
        return cls(model_name_or_path)
