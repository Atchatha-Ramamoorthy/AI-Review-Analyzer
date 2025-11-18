import re, pandas as pd, spacy, nltk
from nltk.corpus import stopwords

# load resources
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords", quiet=True)
stops = set(stopwords.words("english"))

# basic cleaners
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"read more", " ", t)              # remove Flipkart artefact
    t = re.sub(r"http\S+|www\.\S+", " ", t)       # remove links
    t = re.sub(r"[^a-z0-9\s]", " ", t)            # remove emojis/punct
    t = re.sub(r"\s+", " ", t).strip()
    return t

def lemmatize_no_stops(t: str) -> str:
    doc = nlp(t)
    return " ".join([w.lemma_ for w in doc if w.text not in stops])

if __name__ == "__main__":
    df = pd.read_csv("dataset/reviews_raw.csv")
    df["clean"] = df["review"].astype(str).apply(clean_text).apply(lemmatize_no_stops)
    df = df[df["clean"].str.len() > 10].drop_duplicates(subset=["clean"])
    df.to_csv("dataset/reviews_clean.csv", index=False)
    print("Saved dataset/reviews_clean.csv | rows:", len(df))
