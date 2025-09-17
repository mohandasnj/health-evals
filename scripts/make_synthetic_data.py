import json, random, os
os.makedirs("evals/datasets", exist_ok=True)
random.seed(7)
def ex(i):
    return {
        "id": f"ex{i}",
        "rhr": random.randint(55,80),
        "hrv": random.randint(20,80),
        "sleep_efficiency": round(random.uniform(0.6,0.95),2),
        "total_sleep": round(random.uniform(4.5,9.0),1),
        "steps": random.randint(1200,14000),
        "subjective_stress": random.choice(["low","med","high"]),
        "journal_text": random.choice([
            "Felt groggy, short sleep. Lots of meetings.",
            "Slept great. Easy run planned.",
            "Neck tension from laptop; skipped gym.",
            "Late dinner; woke up twice. Could use a nap."
        ])
    }
with open("evals/datasets/test.jsonl","w") as f:
    for i in range(100):
        f.write(json.dumps(ex(i))+"\n")
print("Wrote evals/datasets/test.jsonl")
