"""Phase 1: TF-IDF + logistic regression detector (CPU, about a minute).

Usage:  python scripts/train_detector_tfidf.py
Writes models/detector_tfidf/model.joblib and metrics.json (label 1 = AI).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "detector"
OUT = ROOT / "models" / "detector_tfidf"


def scores(labels, probs):
    return {"accuracy": round(accuracy_score(labels, probs > 0.5), 4),
            "f1": round(f1_score(labels, probs > 0.5), 4),
            "auc": round(roc_auc_score(labels, probs), 4),
            "n": len(labels)}


def main():
    train, val, test = (pd.read_csv(DATA / f"{split}.csv") for split in ["train", "val", "test"])
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True),
        LogisticRegression(C=4, max_iter=2000),
    )
    model.fit(train.text, train.label)

    metrics = {}
    for name, df in [("val", val), ("test", test)]:
        probs = model.predict_proba(df.text)[:, 1]
        metrics[name] = scores(df.label, probs)
        metrics[name]["by_source"] = {source: scores(group.label, probs[group.index])
                                      for source, group in df.groupby("source")}

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for name in ["val", "test"]:
        m = metrics[name]
        print(f"{name:5s} accuracy {m['accuracy']:.3f}  f1 {m['f1']:.3f}  auc {m['auc']:.3f}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
