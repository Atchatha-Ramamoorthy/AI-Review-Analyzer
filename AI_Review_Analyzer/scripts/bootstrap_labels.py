# scripts/bootstrap_labels.py
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
download("vader_lexicon", quiet=True)

df = pd.read_csv("dataset/reviews_clean.csv")
sia = SentimentIntensityAnalyzer()

def sent_label(t):
    s = sia.polarity_scores(t)["compound"]
    return "positive" if s>=0.25 else "negative" if s<=-0.25 else "neutral"

df["sentiment"] = df["clean"].astype(str).apply(sent_label)

# simple seed for authenticity (you will correct)
def seed_auth(t):
    t=t.lower()
    short = len(t.split())<=6
    promo = any(p in t for p in ["must buy","value for money","awesome product","highly recommended","best product"])
    return "fake" if (short or promo) else "genuine"

df["authentic"] = df["clean"].astype(str).apply(seed_auth)

df.to_csv("dataset/reviews_label_seed.csv", index=False)
print("Saved dataset/reviews_label_seed.csv – please open & correct labels.")
