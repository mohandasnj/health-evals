# path shim
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import json, yaml
from statistics import mean
from dotenv import load_dotenv, find_dotenv
from apps.providers import call_chat

load_dotenv(find_dotenv(usecwd=True))
CFG = yaml.safe_load(open("configs/model.yaml"))
RUB = yaml.safe_load(open("configs/judge.yaml"))

PROMPTS = [
"""You are a strict evaluator. Score each dimension 1-5: {dims}.
Return JSON: {{"scores": {keys}, "notes": ""}}.
DATA: {data}
OUTPUT: {out}""",
"""Evaluate per rubric {dims}. Be conservative. JSON only as above.
DATA: {data}
OUTPUT: {out}""",
"""As an expert reviewer, rate 1-5 on {dims}. JSON only (no prose).
DATA: {data}
OUTPUT: {out}"""
]

def dims_text():
    ds = list(RUB["dimensions"].keys())
    return ", ".join(ds), "{" + ", ".join([f'"{d}":X' for d in ds]) + "}"

def judge_one(data, out_text):
    dims, keys = dims_text()
    results = []
    for i in range(CFG["judge"]["n_prompts"]):
        prompt = PROMPTS[i % len(PROMPTS)].format(dims=dims, keys=keys, data=json.dumps(data), out=out_text)
        js_text = call_chat(CFG["judge"], "Return strict JSON only.", prompt)
        start, end = js_text.find("{"), js_text.rfind("}")
        try:
            js = json.loads(js_text[start:end+1]) if start!=-1 and end!=-1 else {"scores":{}}
            scores = js.get("scores", {}) or {d: 3 for d in RUB["dimensions"].keys()}
        except Exception:
            scores = {d: 3 for d in RUB["dimensions"].keys()}
        results.append(scores)
    agg = {k: mean([r[k] for r in results]) for k in results[0].keys()}
    weighted = sum(agg[k]*RUB["weights"][k] for k in agg)
    return {"dim_scores": agg, "final": weighted}

def run(infer_path, out_path):
    outs = []
    for line in open(infer_path):
        ex = json.loads(line)
        j = judge_one(ex["input"], ex["raw"])
        outs.append({"id": ex["id"], "tag": ex["tag"], "blocked": ex["blocked"], **j})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w") as f:
        for o in outs: f.write(json.dumps(o)+"\n")
    print("Judged →", out_path)

if __name__ == "__main__":
    run("out/infer/mut_v1.jsonl", "out/judged/mut_v1.jl")
    run("out/infer/baseline_v2.jsonl", "out/judged/baseline_v2.jl")