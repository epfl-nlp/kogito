import torch
import pytorch_lightning as pl


class GPT2Finetuner(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-5) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, input_ids, mask):
        return self.model(input_ids=input_ids, attention_mask=mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        X = batch
        ids = X["source_ids"]
        mask = X["source_mask"]
        outputs = self.model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch
        ids = X["source_ids"]
        mask = X["source_mask"]
        outputs = self.model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def beam_generations(
    tokenizer,
    model,
    device,
    loader,
    max_length=34,
    top_k=40,
    temperature=1.0,
    top_p=0.9,
    repetition_penalty=1.0,
    num_beams=10,
    num_return_sequences=10,
):
    # This method assumes batch size of 1
    model.eval()
    records = []

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
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

            preds = [
                tokenizer.decode(g, clean_up_tokenization_spaces=True)
                for g in generated_ids
            ]
            try:
                target = [
                    tokenizer.decode(t, clean_up_tokenization_spaces=True) for t in y
                ]
            except Exception:
                target = [""]
            source = [
                tokenizer.decode(s, clean_up_tokenization_spaces=True) for s in ids
            ]

            records.append(
                {"source": source[0], "target": target[0], "generations": preds}
            )

    return records
