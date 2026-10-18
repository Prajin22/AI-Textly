"""Phase 3: make same-content pairs by rewriting human texts into AI style with a local LLM.

HC3/wiki pairs share a topic but not facts. Here Qwen3-1.7B rewrites each *human* text in
AI-assistant style, so the two texts in a pair differ only in style. A pair is kept only if
the meaning survived (MiniLM similarity) and detector v2 reads the rewrite as AI.

Resumable: generations are appended to data/rewrites/raw_<split>.jsonl, and a restarted run
skips what is already there.

Usage:  python scripts/make_rewrite_pairs.py [--train 10000] [--eval 500] [--batch-size 16]
Writes data/humanizer_parallel/{train,val,test}.csv:
    pair_id, source, ai_text, human_text, similarity, ai_prob
"""
import argparse
import ctypes
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from build_dataset import n_words, normalize

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "humanizer"
RAW = ROOT / "data" / "rewrites"
OUT = ROOT / "data" / "humanizer_parallel"
DETECTOR = ROOT / "models" / "detector_roberta"
LLM = "Qwen/Qwen3-1.7B"
SIMILARITY = "sentence-transformers/all-MiniLM-L6-v2"

# Several phrasings, so the AI side is not all one template
STYLES = [
    "Rewrite the following text exactly as ChatGPT would write it: a neutral, polished and formal tone, "
    "a clear opening sentence, smooth transitions such as 'Additionally', 'However' and 'Overall', "
    "and a short concluding sentence.",
    "Rewrite the following text as a helpful AI assistant would explain it: balanced, professional and "
    "well-structured, with generic phrasing, no slang, no personal opinions and no first-person voice.",
    "Rewrite the following text in the typical style of an AI language model: fluent, formal, evenly "
    "paced sentences, abstract wording, and linking phrases such as 'It is important to note' or 'In addition'.",
]
RULES = ("Change the wording and sentence structure throughout; do not copy phrases from the original. "
         "Keep every fact, name and number, and do not add new facts. Keep roughly the same length "
         "(about {words} words). Write plain paragraphs with no title, headings or lists. "
         "Reply with the rewritten text only.")
PREAMBLE = re.compile(r"^(sure|certainly|of course|here is|here's)[^\n]*?:\s*", re.I)
MIN_SIMILARITY, MIN_AI_PROB, LENGTH_RATIO = 0.80, 0.5, (0.6, 1.6)


def build_prompt(tokenizer, text, rng):
    instruction = f"{rng.choice(STYLES)} {RULES.format(words=n_words(text))}\n\nText:\n{text}"
    return tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def clean(text):
    text = PREAMBLE.sub("", text.strip()).replace("**", "").strip().strip('"')
    return normalize(text)


def generate(split, todo, batch_size, seed):
    """Rewrite each human text and append the results to raw_<split>.jsonl."""
    tokenizer = AutoTokenizer.from_pretrained(LLM, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(seed)
    todo = todo.assign(words=todo.human_text.map(n_words)).sort_values("words", ascending=False)
    start, done = time.time(), 0

    with open(RAW / f"raw_{split}.jsonl", "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo.iloc[i:i + batch_size]
            prompts = [build_prompt(tokenizer, text, rng) for text in batch.human_text]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=min(512, int(batch.words.max() * 2.2) + 50),
                                        do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
            texts = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch.itertuples(), texts):
                f.write(json.dumps({"pair_id": row.pair_id, "source": row.source,
                                    "human_text": row.human_text, "ai_text": clean(text)}) + "\n")
            f.flush()
            done += len(batch)
            rate = done / (time.time() - start)
            print(f"[{split}] {done}/{len(todo)}  {rate:.2f} texts/s  ETA {(len(todo) - done) / rate / 60:.0f} min",
                  flush=True)

    del model
    torch.cuda.empty_cache()


def score_and_filter(df):
    """Add meaning similarity and detector-v2 AI probability, then keep the good pairs."""
    embedder = SentenceTransformer(SIMILARITY, device="cuda")
    a = embedder.encode(df.ai_text.tolist(), convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    h = embedder.encode(df.human_text.tolist(), convert_to_tensor=True, normalize_embeddings=True, batch_size=64)
    df["similarity"] = (a * h).sum(-1).cpu().numpy().round(4)

    tokenizer = AutoTokenizer.from_pretrained(DETECTOR)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR, dtype=torch.bfloat16).cuda().eval()
    probs = []
    for i in range(0, len(df), 64):
        inputs = tokenizer(df.ai_text.iloc[i:i + 64].tolist(), truncation=True, max_length=256,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            probs += torch.softmax(detector(**inputs).logits.float(), -1)[:, 1].tolist()
    df["ai_prob"] = [round(p, 4) for p in probs]

    ratio = df.ai_text.map(n_words) / df.human_text.map(n_words)
    keep = (df.similarity >= MIN_SIMILARITY) & (df.ai_prob >= MIN_AI_PROB) & ratio.between(*LENGTH_RATIO)
    return df[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=6000, help="human texts to rewrite from the train split")
    parser.add_argument("--eval", type=int, default=300, help="human texts to rewrite from val and from test")
    parser.add_argument("--batch-size", type=int, default=12, help="lower this if you run out of GPU memory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA not available - install the CUDA build of PyTorch first"
    assert DETECTOR.exists(), "Train detector v2 first: python scripts/train_detector_roberta.py"
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while running, undone on exit
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for split, n in [("val", args.eval), ("test", args.eval), ("train", args.train)]:
        pairs = pd.read_csv(PAIRS / f"{split}.csv").sample(n=n, random_state=args.seed)
        raw_path = RAW / f"raw_{split}.jsonl"
        done = set(pd.read_json(raw_path, lines=True).pair_id) if raw_path.exists() else set()
        todo = pairs[~pairs.pair_id.isin(done)]
        if len(todo):
            generate(split, todo, args.batch_size, args.seed)

    for split in ["train", "val", "test"]:
        raw = pd.read_json(RAW / f"raw_{split}.jsonl", lines=True).drop_duplicates("pair_id")
        kept = score_and_filter(raw)
        kept.to_csv(OUT / f"{split}.csv", index=False)
        print(f"{split:5s} kept {len(kept)}/{len(raw)} pairs  "
              f"(mean similarity {kept.similarity.mean():.3f}, mean AI prob {kept.ai_prob.mean():.3f})")


if __name__ == "__main__":
    main()
