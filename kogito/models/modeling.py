import torch
import os
import wandb
import logging
from tqdm import tqdm
from transformers import Trainer

logger = logging.getLogger("modeling")
device = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerTrainer(Trainer):
    def training_step(self, model, data, *args, **kwargs):
        print("in training step")
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]
        return loss.mean()

    def prediction_step(self, model, data, *args, **kwargs):
        print("in prediction step")
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, labels=ids)
        loss = outputs[0]
        return (loss.mean().detach(), None, None)


def train(
    epoch,
    tokenizer,
    model,
    device,
    loader,
    optimizer,
    val_loader=None,
    output_dir=None,
    log_wandb=False,
):
    model.train()
    batch_count = len(loader)

    for iteration, data in tqdm(enumerate(loader, 0)):
        target_ids = data["target_ids"].to(device, dtype=torch.long)
        input_ids = data["source_ids"].to(device, dtype=torch.long)
        input_mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=input_ids, attention_mask=input_mask, labels=target_ids
        )
        loss = outputs[0]

        if iteration % 100 == 0:
            if log_wandb:
                wandb.log(
                    {
                        "Training Loss": loss.item(),
                        "Epoch": epoch,
                        "Batches left": batch_count - iteration,
                    }
                )
            batches_left = batch_count - iteration
            logger.info(
                f"\nEpoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}, Batches left: {batches_left}"
            )

        if iteration % 500 == 0:
            logger.info(
                f"\nEpoch: {epoch}, Loss:  {loss.item()}, BatchesLeft: {batches_left}"
            )

        if (iteration + 1) % 5000 == 0 and output_dir:
            model.save_pretrained(output_dir + "/iter_{}_model".format(iteration))
            tokenizer.save_pretrained(
                output_dir + "/iter_{}_tokenizer".format(iteration)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0 and val_loader is not None:
            log_eval(
                epoch,
                tokenizer,
                model,
                device,
                val_loader,
                log_wandb=log_wandb,
            )
            model.train()


def log_eval(
    epoch,
    tokenizer,
    model,
    device,
    loader,
    sample_limit=5000,
    log_wandb=False,
):
    model.eval()
    total_loss = 0
    loss_count = 0

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            target_ids = data["target_ids"].to(device, dtype=torch.long)
            input_ids = data["source_ids"].to(device, dtype=torch.long)
            input_mask = data["source_mask"].to(device, dtype=torch.long)

            outputs = model(
                input_ids=input_ids, attention_mask=input_mask, labels=target_ids
            )

            loss = outputs[0]
            total_loss += loss.item()
            loss_count += 1
    if log_wandb:
        wandb.log({"Eval Loss": total_loss / loss_count})
    logger.info("Eval Loss: {}".format(total_loss / loss_count))


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=True,
                max_length=int(os.environ.get("OUT_LEN", 34)),
                num_beams=5,
                top_k=50,
                top_p=0.95,
            )

            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]
            source = [
                tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for s in ids
            ]

            if _ % 100 == 0:
                logger.info(f"Completed {_}")

            sources.extend(source)
            predictions.extend(preds)
            actuals.extend(target)
    return sources, predictions, actuals


def beam_generations(tokenizer, model, device, loader, max_length=34, top_k=40):
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
                temperature=1.0,
                do_sample=False,
                max_length=max_length,
                top_p=0.9,
                top_k=top_k,
                repetition_penalty=1.0,
                num_return_sequences=10 if top_k > 1 else 1,
                num_beams=10,
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

            if _ % 100 == 0:
                logger.info(f"Completed {_}")

    return records
