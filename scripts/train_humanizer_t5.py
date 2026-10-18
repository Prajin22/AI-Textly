"""Phase 4: fine-tune flan-t5-base to turn AI-style text back into the original human wording.

Trains on data/humanizer_parallel (made by make_rewrite_pairs.py): input = AI-style rewrite,
target = the original human text, so the two sides carry the same facts.

Usage:  python scripts/train_humanizer_t5.py [--epochs 3] [--batch-size 4]
Writes models/humanizer_t5/ (model + tokenizer) and metrics.json.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "humanizer_parallel"
OUT = ROOT / "models" / "humanizer_t5"
BASE_MODEL = "google/flan-t5-base"
PREFIX = "Rewrite this in a natural human voice: "
MAX_LEN = 384  # tokens; texts are 40-250 words


def tokenize(tokenizer, batch):
    inputs = tokenizer([PREFIX + t for t in batch["ai_text"]], truncation=True, max_length=MAX_LEN)
    inputs["labels"] = tokenizer(text_target=batch["human_text"], truncation=True, max_length=MAX_LEN)["input_ids"]
    return inputs


def load_split(tokenizer, split):
    """T5's vocabulary lacks some characters; drop pairs that would turn into <unk> tokens."""
    df = pd.read_csv(DATA / f"{split}.csv")
    ds = Dataset.from_pandas(df[["ai_text", "human_text"]]).map(
        lambda batch: tokenize(tokenizer, batch), batched=True)
    has_unk = [tokenizer.unk_token_id in ids or tokenizer.unk_token_id in labels
               for ids, labels in zip(ds["input_ids"], ds["labels"])]
    print(f"{split}: {len(ds)} pairs, dropping {sum(has_unk)} with characters T5 can't represent")
    keep = [i for i, bad in enumerate(has_unk) if not bad]
    return ds.select(keep).remove_columns(["ai_text", "human_text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="lower this if you run out of GPU memory")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA not available - install the CUDA build of PyTorch first"
    assert (DATA / "train.csv").exists(), "Run scripts/make_rewrite_pairs.py first"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train, val = load_split(tokenizer, "train"), load_split(tokenizer, "val")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=max(1, 32 // args.batch_size),  # effective batch of 32
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,  # T5 is unstable in fp16 but fine in bf16
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        disable_tqdm=True,
        dataloader_num_workers=0,  # worker processes are unreliable on Windows
        report_to="none",
        seed=42,
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=val,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                             processing_class=tokenizer)
    trainer.train()

    metrics = {"val_loss": round(trainer.evaluate()["eval_loss"], 4), "train_pairs": len(train),
               "val_pairs": len(val)}
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.rmtree(OUT / "checkpoints", ignore_errors=True)
    print(f"val loss {metrics['val_loss']}  Saved to {OUT}")


if __name__ == "__main__":
    main()
