# -*- coding: utf-8 -*-
"""
Flipkart reviews scraper (robust)

Usage (run from Flaskapp/):
  python flipkart_scraper.py "https://www.flipkart.com/.../product-reviews/ITEM?pid=XXXX" --pages 20 --out data/reviews_iphone.csv --sleep 1.5
"""

import csv, re, time, argparse, random
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

def build_session():
    s = requests.Session()
    # Robust retries for 500/502/503/504
    retries = Retry(total=5, backoff_factor=0.7,
                    status_forcelist=[500, 502, 503, 504],
                    allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Referer": "https://www.flipkart.com/",
        "Connection": "keep-alive",
    })
    return s

def normalize_page_url(base_url: str, page: int) -> str:
    # Ensure we're always appending page= correctly
    if "page=" in base_url:
        return re.sub(r"page=\d+", f"page={page}", base_url)
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}page={page}"

def extract_reviews(html: str):
    """
    Returns list of dicts: {rating, title, review, user, date}
    Handles several current Flipkart layouts.
    """
    soup = BeautifulSoup(html, "html.parser")
    out = []

    # Known wrappers for a single review card
    # (Flipkart changes class names frequently)
    card_selectors = [
        'div[class*="ZmyHeo"]',          # newer
        'div[class*="col"] div[class*="row"]',  # fallback
        'div._27M-vq',                   # older
    ]

    found_cards = []
    for sel in card_selectors:
        cards = soup.select(sel)
        if cards:
            found_cards = cards
            break

    for card in found_cards:
        # Try to pick fields with multiple options
        rating = (
            (card.select_one('div[class*="XQDdHH"]') or card.select_one("._3LWZlK") or card.select_one('[class*="rating"]'))
        )
        title  = card.select_one('p[class*="z9E0IG"]') or card.select_one("p._2-N8zT")
        body   = card.select_one('div[class*="ZmyHeo"] div') or card.select_one("div.t-ZTKy div") or card.select_one("div._6K-7Co") or card.select_one("div._2NsDsF")
        user   = card.select_one('p[class*="MztJPv"]') or card.select_one("p._2sc7ZR._2V5EHH")
        date   = card.find(string=re.compile(r"\d{1,2}\s+\w+\s+\d{4}"))  # approximate

        def clean(node):
            if not node: return ""
            txt = node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node)
            txt = re.sub(r"READ MORE|Read More", "", txt, flags=re.I)
            return txt.strip()

        row = {
            "rating":  clean(rating),
            "title":   clean(title),
            "review":  clean(body),
            "user":    clean(user),
            "date":    clean(date),
        }
        # Keep only non-empty reviews
        if len(row["review"]) > 15:
            out.append(row)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Flipkart ALL REVIEWS url (must contain /product-reviews/ and pid=...)")
    ap.add_argument("--pages", type=int, default=10, help="How many pages to fetch")
    ap.add_argument("--out", default="data/reviews.csv", help="CSV output path (relative to Flaskapp/)")
    ap.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between pages")
    args = ap.parse_args()

    # Sanity checks
    if "/product-reviews/" not in args.url or "pid=" not in args.url:
        print("[error] Please pass the **All Reviews** page URL (contains /product-reviews/ and pid=...)")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s = build_session()
    all_rows = []
    for p in range(1, args.pages + 1):
        url = normalize_page_url(args.url, p)
        try:
            r = s.get(url, timeout=20)
        except requests.RequestException as e:
            print(f"[warn] page {p} -> request failed: {e}")
            break

        if r.status_code != 200:
            print(f"[warn] page {p} -> HTTP {r.status_code}, stopping.")
            break

        rows = extract_reviews(r.text)
        print(f"page {p}: got {len(rows)} reviews")
        all_rows.extend(rows)

        # polite delay with small jitter
        time.sleep(args.sleep + random.uniform(0.3, 0.8))

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rating", "title", "review", "user", "date"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"Saved {len(all_rows)} reviews -> {out_path}")

if __name__ == "__main__":
    main()
