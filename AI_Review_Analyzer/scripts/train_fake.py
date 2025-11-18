import pandas as pd, joblib, numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dataset/reviews_label_seed.csv").dropna(subset=["clean","authentic"])
df["clean"] = df["clean"].astype(str).str.strip()
y = df["authentic"].astype(str)
X_text = df["clean"].astype(str)

counts = Counter(y)
print("Authenticity class counts:", counts)
use_stratify = min(counts.values()) >= 2

vec = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
Xv = vec.fit_transform(X_text)

Xtr, Xte, ytr, yte = train_test_split(
    Xv, y, test_size=0.25, random_state=42, stratify=y if use_stratify else None
)
print("Using stratify:", use_stratify)

# class_weight='balanced' helps when classes are uneven
clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

print("\nAUTHENTICITY REPORT\n", classification_report(yte, pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(yte, pred, labels=sorted(y.unique())))
joblib.dump(clf, "models/fake_model.pkl")
joblib.dump(vec, "models/fake_vectorizer.pkl")
print("Saved → models/fake_model.pkl / fake_vectorizer.pkl")
