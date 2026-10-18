"""Phase 2: fine-tune roberta-base as the AI-vs-human detector (RTX 4050, ~30-60 min).

Usage:  python scripts/train_detector_roberta.py [--epochs 2] [--batch-size 8]
Writes models/detector_roberta/ (model + tokenizer) and metrics.json (label 1 = AI).
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "detector"
OUT = ROOT / "models" / "detector_roberta"
BASE_MODEL = "FacebookAI/roberta-base"
MAX_LEN = 256  # covers ~190 words; training texts are 40-250 words


def scores(labels, probs):
    return {"accuracy": round(accuracy_score(labels, probs > 0.5), 4),
            "f1": round(f1_score(labels, probs > 0.5), 4),
            "auc": round(roc_auc_score(labels, probs), 4),
            "n": len(labels)}


def ai_probs(logits):
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return scores(labels, ai_probs(logits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=8, help="lower this if you run out of GPU memory")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA not available - install the CUDA build of PyTorch first"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    frames = {split: pd.read_csv(DATA / f"{split}.csv") for split in ["train", "val", "test"]}
    datasets = {
        split: Dataset.from_pandas(df[["text", "label"]]).map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
            batched=True, remove_columns=["text"])
        for split, df in frames.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1})
    training_args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=max(1, 32 // args.batch_size),  # effective batch of 32
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        logging_steps=50,
        disable_tqdm=True,
        dataloader_num_workers=0,  # worker processes are unreliable on Windows
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"],
                      eval_dataset=datasets["val"], data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    trainer.train()

    metrics = {}
    for split in ["val", "test"]:
        df = frames[split]
        probs = ai_probs(trainer.predict(datasets[split]).predictions)
        metrics[split] = scores(df.label, probs)
        metrics[split]["by_source"] = {source: scores(group.label, probs[group.index])
                                       for source, group in df.groupby("source")}
        print(f"{split:5s} accuracy {metrics[split]['accuracy']:.4f}  f1 {metrics[split]['f1']:.4f}  "
              f"auc {metrics[split]['auc']:.4f}")

    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.rmtree(OUT / "checkpoints", ignore_errors=True)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
