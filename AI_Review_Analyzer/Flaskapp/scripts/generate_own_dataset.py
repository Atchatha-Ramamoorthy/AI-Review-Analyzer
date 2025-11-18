import random, re, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../Flaskapp
OUT  = ROOT / "data" / "own_reviews_1200.csv"

random.seed(42)

POS_TEMPL = [
    "The {feature} is {adj_pos} and the {benefit} really stands out.",
    "I am {adv_pos} happy with the {feature}; overall a {adj_pos} experience.",
    "Great {feature}! {benefit}. Worth the price.",
    "{brand} nailed it—{feature} feels {adj_pos} and {benefit}.",
    "Battery lasts {duration} and the {feature} is {adj_pos}."
]
NEG_TEMPL = [
    "The {feature} is {adj_neg}; I’m {adv_neg} disappointed.",
    "{brand} failed here—{feature} feels {adj_neg} and {issue}.",
    "Worst experience: {feature} is {adj_neg}, plus {issue}.",
    "{feature} gets {adj_neg} in days; {issue} occurs often.",
    "Overpriced. {feature} is {adj_neg}; {issue} makes it unusable."
]
FAKE_STYLE = [
    "{CLICK} {ALL_CAPS}! Best ever!!! {EMOJI}{EMOJI} Buy now!!!",
    "{ALL_CAPS} DEAL!!! {EMOJI} Never seen before!!! {CLICK}",
    "Amazing product!!! Super super awesome!!! {EMOJI} {CLICK}",
    "Wow!!! Just wow!!! {ALL_CAPS}!!! {CLICK}",
]
REAL_ENDINGS = [
    "Would recommend to friends.", "Good for daily use.",
    "After two weeks, still solid.", "Met my expectations.",
    "If you’re a student, this is fine."
]
ISSUES = ["heats up", "freezes randomly", "battery drains fast", "ghost touches", "lag in camera"]
FEATURES = ["camera", "battery", "display", "speakers", "build quality", "performance"]
BRANDS = ["this phone", "the device", "the handset"]
ADJ_POS = ["excellent", "great", "impressive", "solid", "reliable"]
ADJ_NEG = ["poor", "terrible", "underwhelming", "fragile", "inconsistent"]
ADV_POS = ["really", "extremely", "very"]
ADV_NEG = ["really", "extremely", "quite"]
BENEFITS = ["night photos look clean", "video stabilization helps a lot", "text looks crisp", "audio is loud and clear"]
DURATIONS = ["two days", "a day and half", "almost two days"]

EMOJIS = ["💥","🔥","😍","💯"]
CLICKS = ["Buy now", "Limited offer", "Best deal", "Click fast"]
ALLCAPS = ["MEGA SALE", "BEST PRODUCT", "TOP RATED", "DON'T MISS"]

def gen_pos():
    t = random.choice(POS_TEMPL)
    s = t.format(feature=random.choice(FEATURES),
                 adj_pos=random.choice(ADJ_POS),
                 benefit=random.choice(BENEFITS),
                 brand=random.choice(BRANDS),
                 duration=random.choice(DURATIONS),
                 adv_pos=random.choice(ADV_POS))
    if random.random() < 0.6:
        s += " " + random.choice(REAL_ENDINGS)
    return s

def gen_neg():
    t = random.choice(NEG_TEMPL)
    s = t.format(feature=random.choice(FEATURES),
                 adj_neg=random.choice(ADJ_NEG),
                 issue=random.choice(ISSUES),
                 brand=random.choice(BRANDS),
                 adv_neg=random.choice(ADV_NEG))
    return s

def gen_fake(pos=True):
    base = gen_pos() if pos else gen_neg()
    noisy = f"{base} {random.choice(FAKE_STYLE)}"
    noisy = noisy.replace("{CLICK}", random.choice(CLICKS))
    noisy = noisy.replace("{EMOJI}", random.choice(EMOJIS))
    noisy = noisy.replace("{ALL_CAPS}", random.choice(ALLCAPS))
    return noisy

def gen_real(pos=True):
    return gen_pos() if pos else gen_neg()

def sample_dataset(n=1200):
    rows = []
    # 50% positive real, 20% positive fake, 20% negative real, 10% negative fake
    targets = [
        ("positive","real",0.50),
        ("positive","fake",0.20),
        ("negative","real",0.20),
        ("negative","fake",0.10),
    ]
    counts = [(lab,auth,int(n*ratio)) for (lab,auth,ratio) in targets]
    idx = 1
    for lab,auth,cnt in counts:
        for _ in range(cnt):
            if auth=="real":
                text = gen_real(pos=(lab=="positive"))
            else:
                text = gen_fake(pos=(lab=="positive"))
            # clean a bit
            text = re.sub(r"\s+", " ", text).strip()
            rows.append([idx, text, lab, "genuine" if auth=="real" else "fake"])
            idx += 1
    # adjust if rounding missed a few
    while len(rows) < n:
        rows.append([idx, gen_real(True), "positive", "genuine"]); idx+=1
    return rows

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = sample_dataset(1200)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id","review_text","sentiment","authenticity"])
        w.writerows(rows)
    print(f"Saved {len(rows)} rows -> {OUT}")

if __name__ == "__main__":
    main()
