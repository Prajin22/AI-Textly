# AI TEXTly — Detector + Humanizer Plan

Constraints: free only, everything runs locally on the RTX 4050 (6 GB), one model trained at a time.

| Phase | What | Runs on | Time | Status |
|---|---|---|---|---|
| 0 | Install the CUDA build of PyTorch | — | ~10 min | in progress |
| 1 | Detector v1: TF-IDF + logistic regression, wired into the app | CPU | ~1 min | |
| 2 | Detector v2: fine-tune `roberta-base` on `data/detector` | GPU | ~30–60 min | |
| 3 | Rewrite pairs: `Qwen3-1.7B` rewrites ~10k human texts into AI style, same facts | GPU | ~2–3 h | |
| 4 | Humanizer: fine-tune `flan-t5-base` on (AI-style rewrite → original human text) | GPU | ~1–2 h | |
| 5 | Evaluate + red-team loop: score humanizer output, retrain detector on it | GPU | ~1 h | |
| 6 | App: probability score, Humanize button with before/after score | — | — | |

## Why each phase

1. **Detector v1** replaces the One-Class SVM, which scored 51% (a coin flip) because it never had labels. TF-IDF + logistic regression scores 94.9% on the new test set and needs no GPU.
2. **Detector v2** reads context, not just word counts, so it should hold up better against new AI models and reworded text. RoBERTa is the architecture behind OpenAI's original GPT-2 output detector.
3. **Rewrite pairs.** HC3/wiki pairs share a topic but not facts, so a model trained on them learns to change facts. Rewriting *human* texts into AI style gives pairs where only the style differs. Pairs are kept only if the meaning matches (MiniLM similarity ≥ 0.8) and detector v2 reads the rewrite as AI.
4. **Humanizer** learns the reverse direction: AI-style text → the original human wording.
5. **Red-team loop.** The humanizer is judged on three scores: detector v2's AI probability, an *independent* detector's (v1), and meaning similarity. Then its outputs are labeled AI and added to detector training (v3), and both are re-measured.

## Rules

- Never train on `test.csv` from any phase; splits are by `pair_id`, so a topic never crosses splits.
- Every phase writes its model to `models/<name>/` and its scores to `models/<name>/metrics.json`.
