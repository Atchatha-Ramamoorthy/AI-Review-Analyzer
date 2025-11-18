from pathlib import Path
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

FLASKAPP_DIR = Path(__file__).resolve().parents[1]     # .../Flaskapp
PROJECT_ROOT = FLASKAPP_DIR.parent                     # .../AI_Review_Analyzer
SEED_CSV = FLASKAPP_DIR / "data" / "sentiment_seed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http[s]?://\S+", " ", t)
    # keep punctuation to preserve intensity; just normalize spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    df = pd.read_csv(SEED_CSV)
    df["text"] = df["text"].astype(str).map(clean)
    df = df[df["text"].str.len() >= 3]

    X = df["text"]
    y = df["label"].astype(str)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=20000,
        min_df=1,
        max_df=0.98,
        lowercase=True
    )
    Xtrv = vec.fit_transform(Xtr)
    Xtev = vec.transform(Xte)

    clf = LogisticRegression(
        max_iter=1000,
        C=2.0,
        class_weight="balanced"
    )
    clf.fit(Xtrv, ytr)

    print("\n=== Sentiment (clean) ===")
    print(classification_report(yte, clf.predict(Xtev)))

    joblib.dump(clf, MODELS_DIR / "sentiment_model.pkl")
    joblib.dump(vec, MODELS_DIR / "sentiment_vectorizer.pkl")
    print(f"Saved sentiment model/vectorizer -> {MODELS_DIR}")

if __name__ == "__main__":
    main()
