# path shim
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import argparse, json, yaml, jinja2, pandas as pd, time
import ray
from dotenv import load_dotenv, find_dotenv
from guardrails import Guard
from apps.wellness_coach.schemas import WellnessOutput
from apps.providers import call_chat

def parse_args():
    ap = argparse.ArgumentParser(description="Ray-parallel MUT inference")
    ap.add_argument("--config", default="configs/model.yaml", help="Model config (YAML)")
    ap.add_argument("--split", default="evals/datasets/test.jsonl", help="Input JSONL")
    ap.add_argument("--prompt", default="coach_v1.jinja", help="Prompt template filename")
    ap.add_argument("--tag", default="mut_v1", help="Tag for outputs")
    ap.add_argument("--outdir", default="out/ray/mut_v1", help="Output directory for shards")
    ap.add_argument("--num-actors", type=int, default=4, help="Parallel actors (tune to provider)")
    ap.add_argument("--batch-size", type=int, default=8, help="Rows per Ray batch")
    ap.add_argument("--limit", type=int, default=0, help="Limit examples (0 = all)")
    ap.add_argument("--throttle-sec", type=float, default=0.0, help="Optional sleep per call (useful for hosted APIs)")
    return ap.parse_args()

@ray.remote
class Worker:
    def __init__(self, model_block, prompt_name, throttle_sec=0.0):
        self.block = model_block
        self.throttle = float(throttle_sec)
        self.T = jinja2.Environment(
            loader=jinja2.FileSystemLoader("apps/wellness_coach/prompt_templates")
        ).get_template(prompt_name)
        self.guard = Guard.from_pydantic(WellnessOutput)

    def infer_one(self, row):
        prompt = self.T.render(**row)
        raw = call_chat(self.block, "You are a careful wellness coach.", prompt)
        if self.throttle > 0:
            time.sleep(self.throttle)
        try:
            parsed = self.guard.parse(llm_output=raw, num_reasks=1).validated_output
            blocked = False
        except Exception:
            parsed, blocked = {"summary": "", "suggestions": [], "disclaimer": ""}, True
        return {"id": row["id"], "tag": "mut_v1", "input": row, "raw": raw, "parsed": parsed, "blocked": blocked}

def main():
    load_dotenv(find_dotenv(usecwd=True))
    args = parse_args()
    CFG = yaml.safe_load(open(args.config))

    # Init Ray (local on Mac)
    ray.init(ignore_reinit_error=True)

    # Build dataset
    ds = ray.data.read_json(args.split)
    if args.limit and args.limit > 0:
        ds = ds.limit(args.limit)

    # Spin up actors
    workers = [Worker.remote(CFG["mut"], args.prompt, args.throttle_sec) for _ in range(args.num_actors)]

    # Map in batches for efficiency
    def map_batch(df: pd.DataFrame) -> pd.DataFrame:
        recs = df.to_dict("records")
        futs = [workers[i % len(workers)].infer_one.remote(r) for i, r in enumerate(recs)]
        outs = ray.get(futs)
        return pd.DataFrame(outs)

    os.makedirs(args.outdir, exist_ok=True)
    ds = ds.map_batches(
        map_batch,
        batch_size=args.batch_size,
        batch_format="pandas",
        concurrency=args.num_actors,
        zero_copy_batch=False,
    )
    ds.write_json(args.outdir)  # writes multiple shard files
    print("Ray inference shards →", args.outdir)

if __name__ == "__main__":
    main()
