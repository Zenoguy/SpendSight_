import re
from collections import Counter, defaultdict

import pandas as pd


# ---------- 1. Normalizer (reuse same logic as in regex & MiniLM) ----------
def normalize_desc(desc: str) -> str:
    if not desc:
        return ""

    d = str(desc).upper()

    # Normalize whitespace
    d = re.sub(r"\s+", " ", d)

    # Collapse UPI noise
    d = re.sub(r"UPI/DR/[\w\d/.\-]+/", "UPI/", d)
    d = re.sub(r"UPI/\d+/", "UPI/", d)

    # Strip long numeric IDs
    d = re.sub(r"\d{10,}", " ", d)

    # Collapse double spaces
    d = re.sub(r"\s{2,}", " ", d).strip()

    return d


def main():
    # ---------- 2. Load CSV ----------
    # Adjust this path if needed
    csv_path = "Supabase Snippet Pending Transaction Descriptions.csv"

    df = pd.read_csv(csv_path)

    # Try to find the right column name
    possible_cols = ["description_raw", "description", "desc", "Description_raw"]
    col = None
    for c in possible_cols:
        if c in df.columns:
            col = c
            break

    if col is None:
        print("Could not find description column. Columns are:", df.columns.tolist())
        return

    print(f"[INFO] Using column '{col}' as description source")

    descs = (
        df[col]
        .dropna()
        .astype(str)
    )

    normed = [normalize_desc(x) for x in descs]

    # ---------- 3. Token frequency ----------
    token_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    for d in normed:
        tokens = [t for t in d.split() if len(t) >= 3 and t.isalpha()]
        token_counter.update(tokens)

        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigram_counter[bigram] += 1

        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            trigram_counter[trigram] += 1

    # ---------- 4. UPI handle patterns ----------
    upi_handles = Counter()
    vendors_from_at = Counter()

    upi_pattern = re.compile(r"([A-Z0-9_.-]+@[A-Z]+)")
    for d in normed:
        for handle in upi_pattern.findall(d):
            upi_handles[handle] += 1
            # e.g. ZOMATO@YBL -> vendor "ZOMATO"
            name = handle.split("@")[0]
            vendors_from_at[name] += 1

    # ---------- 5. Print top candidates ----------
    print("\n=== Top 50 tokens (likely vendor / keywords) ===")
    for tok, count in token_counter.most_common(50):
        print(f"{tok:20s} {count}")

    print("\n=== Top 40 bigrams (likely phrase patterns) ===")
    for bg, count in bigram_counter.most_common(40):
        print(f"{bg:35s} {count}")

    print("\n=== Top 20 trigrams (extra phrase context) ===")
    for tg, count in trigram_counter.most_common(20):
        print(f"{tg:50s} {count}")

    print("\n=== Top 30 UPI handles (@xyz) ===")
    for h, count in upi_handles.most_common(30):
        print(f"{h:25s} {count}")

    print("\n=== Top 30 vendors derived from @handles ===")
    for v, count in vendors_from_at.most_common(30):
        print(f"{v:20s} {count}")


if __name__ == "__main__":
    main()
