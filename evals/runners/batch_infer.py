# path shim
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import json, yaml, jinja2
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from guardrails import Guard
from apps.wellness_coach.schemas import WellnessOutput
from apps.providers import call_chat

load_dotenv(find_dotenv(usecwd=True))
CFG = yaml.safe_load(open("configs/model.yaml"))
ENV = jinja2.Environment(loader=jinja2.FileSystemLoader("apps/wellness_coach/prompt_templates"))
guard = Guard.from_pydantic(WellnessOutput)

def guard_and_parse(raw_text: str):
    out = guard.parse(llm_output=raw_text, num_reasks=1)
    if not out.validation_passed:
        raise ValueError("Schema validation failed")
    return out.validated_output

def run(split_path, model_block, prompt_name, tag, out_path, limit=None):
    T = ENV.get_template(prompt_name)
    outs = []
    for i, line in enumerate(open(split_path)):
        if limit and i >= limit: break
        ex = json.loads(line)
        prompt = T.render(**ex)
        raw = call_chat(model_block, "You are a careful wellness coach.", prompt)
        try:
            parsed = guard_and_parse(raw); blocked = False
        except Exception:
            parsed, blocked = {"summary":"","suggestions":[],"disclaimer":""}, True
        outs.append({"id": ex["id"], "tag": tag, "input": ex, "raw": raw, "parsed": parsed, "blocked": blocked})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w") as f:
        for o in outs: f.write(json.dumps(o)+"\n")
    print(f"Wrote {len(outs)} → {out_path}")

if __name__ == "__main__":
    run("evals/datasets/test.jsonl", CFG["mut"], "coach_v1.jinja", "mut_v1", "out/infer/mut_v1.jsonl", limit=5)
    run("evals/datasets/test.jsonl", CFG["baseline"], "coach_v2.jinja", "baseline_v2", "out/infer/baseline_v2.jsonl", limit=5)
