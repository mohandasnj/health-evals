# path shim
import os, sys, json, math, csv, argparse
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from rouge_score import rouge_scorer

# Optional deps (graceful degrade)
try:
    from bert_score import score as bertscore_score
    BERT_OK = True
except Exception:
    BERT_OK = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SBERT_OK = True
except Exception:
    SBERT_OK = False
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    PPL_BASELINE_OK = True
except Exception:
    PPL_BASELINE_OK = False

def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def flatten_from_record(rec):
    p = rec.get("parsed") or {}
    parts = [p.get("summary","")]
    for i, s in enumerate(p.get("suggestions", []) or [], 1):
        parts.append(f"{i}. {s.get('title','')}: {s.get('rationale','')}")
        steps = s.get("steps") or []
        if steps: parts.append("Steps: " + "; ".join(steps))
    parts.append(p.get("disclaimer",""))
    text = "\n".join(t for t in parts if (t or "").strip())
    return text.strip()

def build_ref_map(path):
    return {r["id"]: r["reference_text"] for r in load_jsonl(path)}

def compute_ppl_gpt2(text, model, tok):
    text = (text or "").strip()
    if not text:
        return None
    enc = tok(text, return_tensors="pt")
    # zero-length guard (can happen with weird whitespace-only inputs)
    if enc["input_ids"].numel() == 0:
        return None
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.item())
    try:
        return math.exp(loss)
    except OverflowError:
        return None

def main(mut_in, base_in, refs, outdir, skip_ppl=False, limit=0):
    load_dotenv(find_dotenv(usecwd=True))
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ref_map = build_ref_map(refs)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Lazy init optional models
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") if SBERT_OK else None

    use_ppl = (not skip_ppl) and PPL_BASELINE_OK
    if use_ppl:
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        mdl = GPT2LMHeadModel.from_pretrained("gpt2")
        mdl.eval()

    def eval_file(inpath, name):
        rows = []
        for i, rec in enumerate(load_jsonl(inpath)):
            if limit and i >= limit: break
            rid = rec["id"]
            if rid not in ref_map: 
                continue

            cand = flatten_from_record(rec)
            ref = (ref_map[rid] or "").strip()

            # If either side is empty, fill safe defaults and continue.
            if not cand:
                rows.append({"id": rid, "rougeL_f": 0.0,
                             "bertscore_f1": None,
                             "embed_cosine": None,
                             "ppl_gpt2": None})
                continue
            if not ref:
                # No reference: we can still compute PPL; others need a ref.
                ppl = compute_ppl_gpt2(cand, mdl, tok) if use_ppl else None
                rows.append({"id": rid, "rougeL_f": None,
                             "bertscore_f1": None,
                             "embed_cosine": None,
                             "ppl_gpt2": ppl})
                continue

            # ROUGE-L
            r = scorer.score(ref, cand)["rougeL"].fmeasure

            # BERTScore-F1 (graceful if model missing)
            if BERT_OK:
                try:
                    _, _, f1 = bertscore_score([cand], [ref], lang="en",
                                               model_type="microsoft/deberta-base-mnli",
                                               rescale_with_baseline=True)
                    bsf1 = float(f1[0])
                except Exception:
                    bsf1 = None
            else:
                bsf1 = None

            # Embedding cosine (graceful if model missing)
            if SBERT_OK:
                try:
                    emb = sbert.encode([cand, ref], normalize_embeddings=True)
                    cos = float((emb[0] * emb[1]).sum())
                except Exception:
                    cos = None
            else:
                cos = None

            ppl = compute_ppl_gpt2(cand, mdl, tok) if use_ppl else None

            rows.append({
                "id": rid, "rougeL_f": r,
                "bertscore_f1": bsf1,
                "embed_cosine": cos,
                "ppl_gpt2": ppl
            })

        outcsv = os.path.join(outdir, f"{name}.csv")
        # choose headers robustly
        headers = ["id","rougeL_f","bertscore_f1","embed_cosine","ppl_gpt2"]
        with open(outcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows: w.writerow({k: r.get(k) for k in headers})
        print(f"Wrote {outcsv}  (N={len(rows)})")
        return rows

    eval_file(mut_in, "mut_v1")
    eval_file(base_in, "baseline_v2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mut", default="out/infer/mut_v1.jsonl")
    ap.add_argument("--base", default="out/infer/baseline_v2.jsonl")
    ap.add_argument("--refs", default="evals/datasets/refs.jsonl")
    ap.add_argument("--outdir", default="out/metrics_ref")
    ap.add_argument("--skip-ppl", action="store_true", help="Disable GPT-2 perplexity")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.mut, a.base, a.refs, a.outdir, skip_ppl=a.skip_ppl, limit=a.limit)
