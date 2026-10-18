"""Text helpers shared by the dataset scripts and the app (kept free of heavy imports)."""
import re

SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def n_words(text):
    return len(text.split())


def normalize(text):
    """Remove formatting artifacts that would let a model guess the label without reading the style."""
    text = text.replace("\\n", " ")  # HC3 ChatGPT answers contain a literal "\n"
    text = re.sub(r"\s+", " ", text).strip()
    # Undo the tokenizer spacing in HC3's human answers ("do n't", "word .", "( a )", '" b "')
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r" ([.,!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[$]) ", r"\1", text)
    text = re.sub(r'" ([^"]*?) "', r'"\1"', text)
    # Restore the space lost when newlines were stripped ("markets.However")
    text = re.sub(r"([a-z][.!?])([A-Z])", r"\1 \2", text)
    return text


def split_chunks(text, max_words=180):
    """Split text into chunks of whole sentences, each at most max_words long."""
    chunks, current, count = [], [], 0
    for sentence in SENT_END.split(text):
        words = n_words(sentence)
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks
