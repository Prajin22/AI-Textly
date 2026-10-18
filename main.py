from flask import Flask, render_template, request

from textly_models import Humanizer, load_detector
from textly_text import n_words

app = Flask(__name__, template_folder="template")

# RoBERTa locally; the lightweight TF-IDF model on hosts that cannot fit PyTorch (see textly_models.py)
detector = load_detector()
# The Humanize button only appears once models/humanizer_t5 is trained and PyTorch is available
humanizer = Humanizer() if Humanizer.available() else None

MIN_WORDS = 40  # the models were trained on texts of 40-250 words
MAX_CHARS = 20000  # keeps a request well inside a serverless time limit


def build_verdict(text):
    ai_prob = detector.ai_probability(text)
    if ai_prob >= 0.65:
        kind, label = "ai", "Likely AI-generated"
    elif ai_prob <= 0.35:
        kind, label = "human", "Likely human-written"
    else:
        kind, label = "unsure", "Not sure"
    return {"kind": kind, "label": label, "ai_percent": round(ai_prob * 100),
            "short": n_words(text) < MIN_WORDS}


def humanize_text(text):
    before = round(detector.ai_probability(text) * 100)
    output = humanizer.humanize(text)
    return {"text": output, "before": before, "after": round(detector.ai_probability(output) * 100)}


def render_index(input_text="", error=None, verdict=None, humanized=None):
    return render_template("index.html", input_text=input_text, error=error, verdict=verdict,
                           humanized=humanized, humanizer_ready=humanizer is not None,
                           detector_name=detector.name, max_chars=MAX_CHARS, min_words=MIN_WORDS)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        if not input_text:
            return render_index(error="There is no text given. Please enter some text.")
        if len(input_text) > MAX_CHARS:
            return render_index(input_text=input_text[:MAX_CHARS],
                                error=f"That text is too long. Please keep it under {MAX_CHARS:,} characters.")
        if request.form.get("action") == "humanize" and humanizer is not None:
            return render_index(input_text, humanized=humanize_text(input_text))
        return render_index(input_text, verdict=build_verdict(input_text))
    return render_index()


@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy_policy.html")


@app.route("/terms-and-conditions")
def terms_conditions():
    return render_template("terms_and_conditions.html")


@app.route("/contact-us")
def contact_us():
    return render_template("contact_us.html")


@app.route("/about-us")
def about_us():
    return render_template("about_us.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
