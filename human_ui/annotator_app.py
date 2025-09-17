import os, json, time, csv
import streamlit as st
from pathlib import Path

PAIRS = "out/human/pairs.jsonl"
OUT = "out/human/annotations.csv"

def load_pairs(path):
    return [json.loads(l) for l in open(path)]

def main():
    st.set_page_config(page_title="Human Eval", layout="wide")
    st.title("Human Evaluation – Pairwise & Likert")

    annotator = st.text_input("Annotator name (required):")
    if not annotator:
        st.stop()

    pairs = load_pairs(PAIRS)
    idx = st.number_input("Item index", min_value=0, max_value=len(pairs)-1, value=0, step=1)
    ex = pairs[idx]

    with st.expander("Input ID", expanded=True):
        st.code(ex["id"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System A")
        st.text_area("A output", ex["systemA"]["text"], height=380)
    with col2:
        st.subheader("System B")
        st.text_area("B output", ex["systemB"]["text"], height=380)

    st.markdown("---")
    pref = st.radio("Pairwise preference", ["A","B","Tie"], horizontal=True)
    st.markdown("### Likert ratings (1=poor, 5=excellent)")
    c1,c2,c3,c4 = st.columns(4)
    with c1: helpful = st.slider("Helpfulness", 1,5,3)
    with c2: factual = st.slider("Factuality", 1,5,3)
    with c3: safety  = st.slider("Safety", 1,5,4)
    with c4: clarity = st.slider("Clarity", 1,5,4)
    tags = st.multiselect("Error types", ["Hallucination","Unsafe/Medical advice","Off-topic","Incoherent/format","No disclaimer","Other"])
    notes = st.text_area("Notes (optional)", "")

    if st.button("Save"):
        Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts","annotator","id","pref","helpful","factual","safety","clarity","tags","notes"])
            w.writerow([int(time.time()), annotator, ex["id"], pref, helpful, factual, safety, clarity, "|".join(tags), notes])
        st.success("Saved!")
        st.balloons()

if __name__ == "__main__":
    main()
