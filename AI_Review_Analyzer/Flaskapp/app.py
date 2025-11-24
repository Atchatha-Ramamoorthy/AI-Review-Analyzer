from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import joblib
import logging
import numpy as np   
import csv
from datetime import datetime
from collections import Counter
import re



app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


APP_DIR = Path(__file__).resolve().parent              
BASE_DIR = APP_DIR.parent                              
MODELS_DIR = BASE_DIR / "models"                       
HISTORY_FILE = APP_DIR / "data" / "review_history.csv" 


def load_or_die(path: Path, name: str):
    """Load a model or raise a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    log.info("Loaded %s from %s", name, path)
    return joblib.load(path)


# Load all four models/vectorizers once when the app starts
try:
    log.info("DEBUG MODELS_DIR: %s", MODELS_DIR)
    log.info("DEBUG FILES IN MODELS: %s", [p.name for p in MODELS_DIR.glob("*.pkl")])

    SENTIMENT_MODEL = load_or_die(MODELS_DIR / "sentiment_model.pkl", "sentiment model")
    SENTIMENT_VECT = load_or_die(
        MODELS_DIR / "sentiment_vectorizer.pkl", "sentiment vectorizer"
    )

    FAKE_MODEL = load_or_die(MODELS_DIR / "fake_model.pkl", "fake/genuine model")
    FAKE_VECT = load_or_die(
        MODELS_DIR / "fake_vectorizer.pkl", "fake/genuine vectorizer"
    )
except Exception as e:
    log.exception("MODEL LOAD ERROR: %s", e)
    # Stop the app if models are missing
    raise

def predict_sentiment(review_text: str):
    vec = SENTIMENT_VECT.transform([review_text])
    probs = SENTIMENT_MODEL.predict_proba(vec)[0]
    pred_idx = np.argmax(probs)
    label = SENTIMENT_MODEL.classes_[pred_idx]
    sentiment = "Positive" if label == "positive" else "Negative"
    return sentiment, float(probs[pred_idx] * 100)


def predict_authenticity(review_text: str):
    vec = FAKE_VECT.transform([review_text])
    probs = FAKE_MODEL.predict_proba(vec)[0]
    pred_idx = np.argmax(probs)
    label = FAKE_MODEL.classes_[pred_idx]
    authenticity = "Genuine" if label == "genuine" else "Fake"
    return authenticity, float(probs[pred_idx] * 100)

def log_review(result: dict) -> None:
    """Append one analyzed review to data/review_history.csv."""
    first_time = not HISTORY_FILE.exists()

    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header first time
        if first_time:
            writer.writerow([
                "timestamp",
                "review",
                "sentiment",
                "sentiment_prob",
                "authenticity",
                "authenticity_prob",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            result["review"],
            result["sentiment"],
            f"{result['sentiment_prob']:.1f}",
            result["authenticity"],
            f"{result['authenticity_prob']:.1f}",
        ])

STOPWORDS = {
    "the","a","an","is","am","are","was","were","and","or","of","to","in",
    "it","this","that","for","on","with","as","at","by","from","very",
    "i","we","you","he","she","they","my","our","your","their","but",
    "so","just","too","also","if","be","have","has","had","its"
}

def tokenize(text: str):
    """Lowercase + simple word split + stopword removal."""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

@app.route("/", methods=["GET"])
def home():
    """Main page with big textarea + Analyze button."""
    return render_template("home.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    """
    Handle form submission from home.html
    GET  -> redirect back to home
    POST -> run both models and show the result page
    """
    if request.method == "GET":
        return redirect(url_for("home"))

    review = request.form.get("review", "").strip()

    if not review:
        # No text typed -> re-render home with error message
        return render_template(
            "home.html", error="Please enter a review before analyzing."
        )
    


    sentiment, sentiment_prob = predict_sentiment(review)
    authenticity, authenticity_prob = predict_authenticity(review)

    result = {
    "review": review,
    "sentiment": sentiment,
    "sentiment_prob": sentiment_prob,
    "authenticity": authenticity,
    "authenticity_prob": authenticity_prob,
    }

    # save to history log
    log_review(result)

    return render_template("ana.html", result=result)

    # ana.html is the result page
    return render_template("ana.html", result=result)


@app.route("/about", methods=["GET"])
def about():
    """About / project description page."""
    return render_template("about.html")


@app.route("/how_it_works", methods=["GET"])
def how_it_works():
    """Explain pipeline: Input -> NLP/ML -> Output."""
    return render_template("how_it_works.html")


@app.route("/history", methods=["GET"])
def history():
    rows = []
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            rows.reverse()   # show latest first

    return render_template("history.html", rows=rows)

@app.route("/word_cloud", methods=["GET"])
def word_cloud():
    """
    Build simple 'word cloud' style insights from the history:
    - words that appear often in Positive vs Negative reviews
    - words that appear often in Genuine vs Fake reviews
    """
    pos_counter = Counter()
    neg_counter = Counter()
    gen_counter = Counter()
    fake_counter = Counter()

    if HISTORY_FILE.exists():
        with HISTORY_FILE.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                review_text = row.get("review", "")
                sentiment = row.get("sentiment", "")
                authenticity = row.get("authenticity", "")

                words = tokenize(review_text)

                # sentiment words
                if sentiment == "Positive":
                    pos_counter.update(words)
                elif sentiment == "Negative":
                    neg_counter.update(words)

                # authenticity words
                if authenticity == "Genuine":
                    gen_counter.update(words)
                elif authenticity == "Fake":
                    fake_counter.update(words)

    def make_cloud(counter: Counter, max_words: int = 30):
        items = counter.most_common(max_words)
        if not items:
            return []
        max_count = items[0][1]
        cloud = []
        for word, cnt in items:
            ratio = cnt / max_count
            if ratio > 0.66:
                size = "large"
            elif ratio > 0.33:
                size = "medium"
            else:
                size = "small"
            cloud.append({"word": word, "count": cnt, "size": size})
        return cloud

    pos_words = make_cloud(pos_counter)
    neg_words = make_cloud(neg_counter)
    gen_words = make_cloud(gen_counter)
    fake_words = make_cloud(fake_counter)

    return render_template(
        "word_cloud.html",
        pos_words=pos_words,
        neg_words=neg_words,
        gen_words=gen_words,
        fake_words=fake_words,
    )


@app.route("/bulk", methods=["GET"])
def bulk():
    """(Optional) Bulk upload / history page."""
    return render_template("bulk.html")


# -------------------------------------------------
# Run the app
# -------------------------------------------------
if __name__ == "__main__":
    # Run from inside the Flaskapp folder:
    #   cd C:\Users\fcgoa\AI_Review_Analyzer\Flaskapp
    #   python app.py
    app.run(debug=True)
