import csv, statistics as st
from collections import Counter, defaultdict

def main(path="out/human/annotations.csv"):
    rows=list(csv.DictReader(open(path)))
    prefs=Counter(r["pref"] for r in rows)
    print("Pairwise:", dict(prefs))
    for dim in ["helpful","factual","safety","clarity"]:
        vals=[int(r[dim]) for r in rows]
        print(f"{dim}: mean={st.mean(vals):.2f}, sd={st.pstdev(vals):.2f}")
    by_id = defaultdict(list)
    for r in rows: by_id[r["id"]].append(r["pref"])
    # simple win-rate (mut over baseline) using hidden answer key is possible if you join with pairs
    print(f"N annotations: {len(rows)}; N items: {len(by_id)}")

if __name__ == "__main__":
    main()
