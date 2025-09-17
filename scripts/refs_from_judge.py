# path shim
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import yaml, jinja2
from guardrails import Guard
from apps.wellness_coach.schemas import WellnessOutput
from apps.providers import call_chat

def flatten(parsed):
    if not parsed: return ""
    parts = [parsed.get("summary","")]
    for i, s in enumerate(parsed.get("suggestions", []), 1):
        parts.append(f"{i}. {s.get('title','')}: {s.get('rationale','')}")
        steps = s.get("steps", [])
        if steps: parts.append("Steps: " + "; ".join(steps))
    parts.append(parsed.get("disclaimer",""))
    return "\n".join(p for p in parts if p)

def main(infile="evals/datasets/test.jsonl", outpath="evals/datasets/refs.jsonl", limit=50):
    load_dotenv(find_dotenv(usecwd=True))
    CFG = yaml.safe_load(open("configs/model.yaml"))
    T = jinja2.Environment(
        loader=jinja2.FileSystemLoader("apps/wellness_coach/prompt_templates")
    ).get_template("coach_v1.jinja")

    guard = Guard.from_pydantic(WellnessOutput)
    Path(os.path.dirname(outpath)).mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as out:
        for i, line in enumerate(open(infile)):
            if limit and i >= limit: break
            ex = json.loads(line)
            prompt = T.render(**ex)
            raw = call_chat(CFG["judge"], "You are an expert wellness coach.", prompt)
            try:
                parsed = guard.parse(llm_output=raw, num_reasks=1).validated_output
            except Exception:
                parsed = None
            out.write(json.dumps({"id": ex["id"], "reference_text": flatten(parsed)}) + "\n")
    print(f"Wrote references → {outpath}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="evals/datasets/test.jsonl")
    ap.add_argument("--out", default="evals/datasets/refs.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()
    main(a.infile, a.out, a.limit)
