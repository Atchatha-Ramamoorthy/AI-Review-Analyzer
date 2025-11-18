import pandas as pd, joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dataset/reviews_label_seed.csv").dropna(subset=["clean","sentiment"])
df["clean"] = df["clean"].astype(str).str.strip()
y_raw = df["sentiment"].astype(str)
X = df["clean"].astype(str)

# --- fallback: make it binary if any class < 2 ---
counts = Counter(y_raw)
print("Sentiment class counts (original):", counts)

if len([c for c in counts.values() if c >= 2]) < 2:
    print("Not enough classes for 3-way. Falling back to binary: positive vs nonpositive.")
    y = y_raw.apply(lambda s: "positive" if s == "positive" else "negative")
else:
    y = y_raw.copy()

counts2 = Counter(y)
print("Training counts:", counts2)
use_stratify = min(counts2.values()) >= 2

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y if use_stratify else None
)
print("Using stratify:", use_stratify)

vec = TfidfVectorizer(max_features=12000, ngram_range=(1,2))
Xtrv, Xtev = vec.fit_transform(Xtr), vec.transform(Xte)

clf = LogisticRegression(max_iter=600, class_weight="balanced")
clf.fit(Xtrv, ytr)
pred = clf.predict(Xtev)

print("\nSENTIMENT REPORT\n", classification_report(yte, pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(yte, pred, labels=sorted(y.unique())))

joblib.dump(clf, "models/sentiment_model.pkl")
joblib.dump(vec, "models/sentiment_vectorizer.pkl")
print("Saved → models/sentiment_model.pkl / sentiment_vectorizer.pkl")
