import os, sys, json, glob

def main(indir, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    n = 0
    with open(outfile, "w") as out:
        for fp in sorted(glob.glob(os.path.join(indir, "*.json"))):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Ray writes JSON lines; keep as-is
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out.write(json.dumps(obj) + "\n")
                    n += 1
    print(f"Merged {n} lines → {outfile}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/collect_ray_outputs.py <indir> <outfile>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
