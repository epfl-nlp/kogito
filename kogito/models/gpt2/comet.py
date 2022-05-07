from typing import Optional
import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments
import wandb

from kogito.models.modeling import beam_generations, TransformerTrainer
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
        batch_size: int = 2,
        in_len: int = 16,
        out_len: int = 34,
        summary_len: int = 0,
        epochs: int = 3,
        lr_rate: float = 1e-5,
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
            lr_rate (float, optional): Learning rate. Defaults to 1e-5.
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
            summ_len=summary_len,
        )
        val_dataset = KnowledgeDataset(
            val_graph,
            tokenizer=self.tokenizer,
            source_len=in_len,
            summ_len=out_len - in_len,
            is_eval=True,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr_rate)

        trainer_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr_rate,
            num_train_epochs=epochs,
            dataloader_drop_last=True,
            report_to=None,
            gradient_checkpointing=True,
            gradient_accumulation_steps=2,
        )
        trainer = TransformerTrainer(
            model=self.model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            optimizers=(optimizer, None),
        )
        trainer.train()
        # train_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
        # val_params = {"batch_size": 1, "shuffle": False, "num_workers": 0}

        # train_loader = DataLoader(train_dataset, **train_params, drop_last=True)
        # val_loader = DataLoader(val_dataset, **val_params, drop_last=True)

        if log_wandb:
            config = {
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr_rate,
                "seed": seed,
                "in_len": in_len,
                "summary_len": summary_len,
                "out_len": out_len,
            }
            wandb.init(project="kogito_comet_gpt2", config=config)

        # for epoch in range(epochs):
        #     train(
        #         epoch,
        #         self.tokenizer,
        #         self.model,
        #         device,
        #         train_loader,
        #         optimizer,
        #         val_loader,
        #         log_wandb=log_wandb,
        #         output_dir=output_dir,
        #     )
        #     self.save_pretrained(f"{output_dir}/checkpoint_{epoch}")

        self.save_pretrained(f"{output_dir}/final")

        return self.model

    def generate(
        self,
        input_graph: KnowledgeGraph,
        in_len: int = 16,
        out_len: int = 34,
        top_k: int = 1,
    ) -> KnowledgeGraph:
        """Generate inferences from knowledge model

        Args:
            input_graph (KnowledgeGraph): Input dataset
            in_len (int, optional): Input length. Defaults to 16.
            out_len (int, optional): Output length. Defaults to 34.
            top_k (int, optional): Top k inferences to consider. Defaults to 1.

        Returns:
            KnowledgeGraph: Completed knowledge graph
        """
        params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
        dataset = KnowledgeDataset(
            input_graph,
            tokenizer=self.tokenizer,
            source_len=in_len,
            summ_len=out_len - in_len,
            model="gpt2",
            is_eval=True,
        )
        loader = DataLoader(dataset, **params, drop_last=False)
        generations = beam_generations(
            self.tokenizer, self.model, device, loader, top_k=top_k
        )
        outputs = []
        for input_kg, gen in zip(input_graph, generations):
            output_kg = input_kg.copy()
            output_kg.tails = gen["generations"]
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
