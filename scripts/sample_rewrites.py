"""Quality check before the full Phase 3 run: rewrite a few human texts and print them side by side.

Usage:  PYTHONPATH=scripts python scripts/sample_rewrites.py [--n 20]
Writes only to logs/sample_rewrites.csv.
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import n_words
from make_rewrite_pairs import LLM, LENGTH_RATIO, MIN_AI_PROB, MIN_SIMILARITY, build_prompt, clean, score_and_filter

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    sample = pd.read_csv(ROOT / "data" / "humanizer" / "val.csv").sample(n=args.n, random_state=7).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(LLM, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16, device_map="cuda").eval()
    rng, rewrites, start = random.Random(7), [], time.time()

    for i in range(0, len(sample), args.batch_size):
        batch = sample.iloc[i:i + args.batch_size]
        prompts = [build_prompt(tokenizer, text, rng) for text in batch.human_text]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        words = batch.human_text.map(n_words).max()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=min(512, int(words * 2.2) + 50),
                                 do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        rewrites += [clean(t) for t in tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)]
    seconds = time.time() - start
    print(f"Generated {len(sample)} rewrites in {seconds:.0f}s ({len(sample) / seconds:.2f} texts/s, "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)")

    del model
    torch.cuda.empty_cache()
    df = sample[["pair_id", "source", "human_text"]].assign(ai_text=rewrites)
    kept = score_and_filter(df)  # also adds similarity and ai_prob columns to df for every row
    df["kept"] = df.pair_id.isin(kept.pair_id)
    df.to_csv(ROOT / "logs" / "sample_rewrites.csv", index=False)
    print(f"Kept {len(kept)}/{len(df)} (similarity >= {MIN_SIMILARITY}, AI prob >= {MIN_AI_PROB}, "
          f"length ratio {LENGTH_RATIO})")
    print(df[["similarity", "ai_prob"]].describe().loc[["mean", "min", "50%", "max"]].round(3))


if __name__ == "__main__":
    main()
