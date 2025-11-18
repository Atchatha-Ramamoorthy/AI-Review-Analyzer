from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import re

FLASKAPP_DIR = Path(__file__).resolve().parents[1]      # .../Flaskapp
PROJECT_ROOT = FLASKAPP_DIR.parent                      # .../AI_Review_Analyzer
DATA = FLASKAPP_DIR / "data" / "own_reviews_1200.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def basic_clean(t: str) -> str:
    t = re.sub(r"http[s]?://\S+", " ", t or "")
    t = re.sub(r"[^A-Za-z0-9(),.!?'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_data():
    df = pd.read_csv(DATA)
    df["review_text"] = df["review_text"].astype(str).map(basic_clean)
    return df

def train_sentiment(df):
    X = df["review_text"]
    y = df["sentiment"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
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
    print("\n=== Sentiment ===")
    print(classification_report(yte, clf.predict(Xtev)))
    joblib.dump(clf, MODELS_DIR / "sentiment_model.pkl")
    joblib.dump(vec, MODELS_DIR / "sentiment_vectorizer.pkl")

def train_auth(df):
    X = df["review_text"]
    y = df["authenticity"]  # 'genuine' or 'fake'
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    Xtrv = vec.fit_transform(Xtr)
    Xtev = vec.transform(Xte)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=None)
    clf.fit(Xtrv, ytr)
    print("\n=== Authenticity ===")
    print(classification_report(yte, clf.predict(Xtev)))
    joblib.dump(clf, MODELS_DIR / "fake_model.pkl")
    joblib.dump(vec, MODELS_DIR / "fake_vectorizer.pkl")

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows from {DATA}")
    train_sentiment(df)
    train_auth(df)
    print(f"\nSaved models to {MODELS_DIR}")
