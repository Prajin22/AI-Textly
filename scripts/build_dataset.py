"""Build the human-vs-AI datasets for the detector and the humanizer.

Both sources pair a human text with an AI text written for the same question/topic:
  - HC3 (Hello-SimpleAI/HC3, CC-BY-SA-4.0): human vs ChatGPT answers to ~24k
    questions from Reddit ELI5, finance, medicine, open QA and Wikipedia CS/AI.
  - GPT-wiki-intro (aadityaubhat/GPT-wiki-intro, CC): real Wikipedia intros vs
    GPT-3 (Curie) intros for 150k topics.

Outputs (label: 0 = human, 1 = AI), split by pair so a topic never leaks across splits:
  data/detector/{train,val,test}.csv    text, label, source, pair_id
  data/humanizer/{train,val,test}.csv   pair_id, source, prompt, ai_text, human_text

Usage:  python scripts/build_dataset.py [--wiki-pairs 25000] [--seed 42]
"""
import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textly_text import SENT_END, n_words, normalize  # noqa: E402  (shared with the app)
RAW = ROOT / "data" / "raw"
SOURCES = {
    "HC3_all.jsonl": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl",
    "GPT-wiki-intro.csv.zip": "https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/resolve/main/GPT-wiki-intro.csv.zip",
}
MIN_WORDS, MAX_WORDS = 40, 250
REFUSAL = re.compile(r"as an ai|i'm sorry, but|i am sorry, but|i'm not able to|i am not able to|i'm unable to", re.I)


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (RAW / name).exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, RAW / name)


def trim(text, max_words):
    """Keep whole sentences up to max_words and drop a trailing unfinished sentence."""
    out, n = [], 0
    for sent in SENT_END.split(text):
        if out and n + n_words(sent) > max_words:
            break
        out.append(sent)
        n += n_words(sent)
    if len(out) > 1 and not out[-1].rstrip("\"')").endswith((".", "!", "?")):
        out.pop()
    return " ".join(out)


def make_pair(human, ai):
    """Normalize both texts and trim them to the same length, so length is not a giveaway."""
    human, ai = normalize(human), normalize(ai)
    if REFUSAL.search(ai):
        return None
    target = min(n_words(human), n_words(ai), MAX_WORDS)
    if target < MIN_WORDS:
        return None
    human, ai = trim(human, target), trim(ai, target)
    if min(n_words(human), n_words(ai)) < MIN_WORDS:
        return None
    return human, ai


def hc3_pairs():
    with open(RAW / "HC3_all.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for i, row in enumerate(rows):
        if not row["human_answers"] or not row["chatgpt_answers"]:
            continue
        human = max(row["human_answers"], key=n_words)
        pair = make_pair(human, row["chatgpt_answers"][0])
        if pair:
            yield dict(pair_id=f"hc3-{i}", source=f"hc3_{row['source']}",
                       prompt=normalize(row["question"]), human_text=pair[0], ai_text=pair[1])


def wiki_pairs(limit, seed):
    df = pd.read_csv(RAW / "GPT-wiki-intro.csv.zip").sample(frac=1, random_state=seed)
    count = 0
    for row in df.itertuples():
        if count >= limit:
            break
        pair = make_pair(row.wiki_intro, row.generated_intro)
        if pair:
            count += 1
            yield dict(pair_id=f"wiki-{row.id}", source="wiki_intro",
                       prompt=row.title, human_text=pair[0], ai_text=pair[1])


def assign_splits(pairs, seed):
    """80/10/10 split per source, by pair, so a question's human and AI texts stay together."""
    rng = random.Random(seed)
    for source in pairs.source.unique():
        idx = pairs.index[pairs.source == source].tolist()
        rng.shuffle(idx)
        n_train, n_val = int(len(idx) * 0.8), int(len(idx) * 0.1)
        pairs.loc[idx[:n_train], "split"] = "train"
        pairs.loc[idx[n_train:n_train + n_val], "split"] = "val"
        pairs.loc[idx[n_train + n_val:], "split"] = "test"
    return pairs


def report(detector):
    print("\nRows per split and source:")
    print(detector.pivot_table(index="source", columns="split", values="text", aggfunc="count").fillna(0).astype(int))
    print("\nMedian words (human / AI) per source:")
    words = detector.assign(words=detector.text.map(n_words))
    print(words.pivot_table(index="source", columns="label", values="words", aggfunc="median"))
    print("\nArtifact rates after cleaning (human / AI) - should be close to each other:")
    for name, pattern in [("space before . or ,", r" [.,]"), ("tokenized n't", r" n't"), ("literal \\n", r"\\n")]:
        rates = detector.groupby("label").text.apply(lambda s: s.str.contains(pattern, regex=True).mean())
        print(f"  {name:22s} {rates[0]:.4f} / {rates[1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wiki-pairs", type=int, default=25000, help="pairs to take from GPT-wiki-intro (max ~150k)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download()
    pairs = pd.DataFrame([*hc3_pairs(), *wiki_pairs(args.wiki_pairs, args.seed)])
    pairs = pairs.drop_duplicates("human_text").drop_duplicates("ai_text").reset_index(drop=True)
    pairs = assign_splits(pairs, args.seed)

    detector = pd.concat([
        pairs.assign(text=pairs.human_text, label=0),
        pairs.assign(text=pairs.ai_text, label=1),
    ])[["text", "label", "source", "pair_id", "split"]]

    for name, df, cols in [("detector", detector, ["text", "label", "source", "pair_id"]),
                           ("humanizer", pairs, ["pair_id", "source", "prompt", "ai_text", "human_text"])]:
        out = ROOT / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            part = df[df.split == split].sample(frac=1, random_state=args.seed)
            part[cols].to_csv(out / f"{split}.csv", index=False)

    print(f"Built {len(pairs)} human/AI pairs -> {len(detector)} detector rows")
    report(detector)


if __name__ == "__main__":
    main()
