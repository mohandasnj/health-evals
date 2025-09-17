import os, json, random
from pathlib import Path

def flatten(rec):
    p = rec.get("parsed") or {}
    parts = [p.get("summary","")]
    for i, s in enumerate(p.get("suggestions", []), 1):
        parts.append(f"{i}. {s.get('title','')}: {s.get('rationale','')}")
        steps = s.get("steps", [])
        if steps: parts.append("Steps: " + "; ".join(steps))
    parts.append(p.get("disclaimer",""))
    return "\n".join(t for t in parts if t)

def index_by_id(path):
    d={}
    for line in open(path):
        r=json.loads(line)
        d[r["id"]]=r
    return d

def main(mut="out/infer/mut_v1.jsonl", base="out/infer/baseline_v2.jsonl", out="out/human/pairs.jsonl"):
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    M=index_by_id(mut); B=index_by_id(base)
    with open(out,"w") as f:
        for k in sorted(set(M).intersection(B)):
            A_text, B_text = flatten(M[k]), flatten(B[k])
            # blind & randomize order
            pair = [("A", A_text, "mut_v1"), ("B", B_text, "baseline_v2")]
            random.shuffle(pair)
            f.write(json.dumps({
                "id": k,
                "systemA": {"text": pair[0][1]},
                "systemB": {"text": pair[1][1]},
                "answer_key": { "A_is": pair[0][2], "B_is": pair[1][2] }
            })+"\n")
    print(f"Wrote tasks → {out}")

if __name__ == "__main__":
    main()
