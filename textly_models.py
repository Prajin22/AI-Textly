"""Runtime models for the app.

Locally the app uses the fine-tuned RoBERTa detector (and the flan-t5 humanizer once trained). On a small
host such as Vercel, where PyTorch and a 480 MB model do not fit, it falls back to the TF-IDF detector.
"""
from importlib.util import find_spec
from pathlib import Path

from textly_text import n_words, normalize, split_chunks

MODELS = Path(__file__).resolve().parent / "models"
HUMANIZER_PREFIX = "Rewrite this in a natural human voice: "


def _weighted_mean(chunks, probs):
    return sum(p * n_words(c) for c, p in zip(chunks, probs)) / sum(n_words(c) for c in chunks)


class TfidfDetector:
    name = "TF-IDF + logistic regression"

    def __init__(self):
        import joblib
        self.model = joblib.load(MODELS / "detector_tfidf" / "model.joblib")

    def ai_probability(self, text):
        chunks = split_chunks(normalize(text))
        return float(_weighted_mean(chunks, self.model.predict_proba(chunks)[:, 1]))


class RobertaDetector:
    name = "RoBERTa"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = MODELS / "detector_roberta"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()

    def ai_probability(self, text):
        """Probability that the text is AI-written: the word-weighted average over its chunks."""
        chunks = split_chunks(normalize(text))
        inputs = self.tokenizer(chunks, truncation=True, max_length=256, padding=True,
                                return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**inputs).logits.float(), dim=-1)[:, 1].cpu().tolist()
        return float(_weighted_mean(chunks, probs))


def load_detector():
    """RoBERTa if its weights and PyTorch are available, otherwise the lightweight TF-IDF model."""
    if find_spec("torch") and find_spec("transformers") and (MODELS / "detector_roberta" / "model.safetensors").exists():
        return RobertaDetector()
    return TfidfDetector()


class Humanizer:
    """Loaded only if PyTorch and models/humanizer_t5 exist (Phase 4)."""

    @staticmethod
    def available():
        return bool(find_spec("torch") and find_spec("transformers")
                    and (MODELS / "humanizer_t5" / "model.safetensors").exists())

    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = MODELS / "humanizer_t5"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device).eval()

    def humanize(self, text):
        chunks = split_chunks(normalize(text))
        inputs = self.tokenizer([HUMANIZER_PREFIX + c for c in chunks], truncation=True, max_length=384,
                                padding=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=400, num_beams=4, no_repeat_ngram_size=4,
                                         repetition_penalty=1.1)
        return " ".join(self.tokenizer.batch_decode(output, skip_special_tokens=True))
