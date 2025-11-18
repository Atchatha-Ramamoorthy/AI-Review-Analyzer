import requests, time
from bs4 import BeautifulSoup
import pandas as pd

def fetch_reviews(base_url, pages=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": "https://www.flipkart.com/",
    }
    all_reviews = []
    for p in range(1, pages + 1):
        url = f"{base_url}&page={p}" if "?" in base_url else f"{base_url}?page={p}"
        r = requests.get(url, headers=headers, timeout=25)
        if r.status_code != 200:
            print("Skip page", p, "status:", r.status_code)
            continue
        soup = BeautifulSoup(r.text, "html.parser")

        # ✅ Updated selector (we found this works)
        blocks = soup.select("div.ZmyHeo")

        for b in blocks:
            txt = b.get_text(" ", strip=True)
            if txt and len(txt) > 20:
                all_reviews.append(txt)
        print(f"Page {p}: grabbed {len(blocks)} reviews, total={len(all_reviews)}")
        time.sleep(1.2)

    return all_reviews


if __name__ == "__main__":
    BASE = "https://www.flipkart.com/analogue-minimalist-slim-series-smart-strap-clip-soft-silicon-boys-analog-watch-men/product-reviews/itm9d0aece715efd?pid=WATGZV5UZZ4EZZ64&lid=LSTWATGZV5UZZ4EZZ64NSUW0C&marketplace=FLIPKART"
    reviews = fetch_reviews(BASE, pages=6)
    pd.DataFrame({"review": reviews}).to_csv("dataset/reviews_raw.csv", index=False, encoding="utf-8")
    print(f"\n✅ Saved dataset/reviews_raw.csv | Total reviews: {len(reviews)}")
