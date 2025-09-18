# Health-Evals: Scalable Evaluation Harness for Wellness Coaching LLMs

**A reproducible evaluation system** for comparing wellness-coaching LLMs end-to-end with comprehensive metrics and human evaluation capabilities.

## ✨ Key Features

- **A/B model comparisons** (LLM vs Baseline) with **schema-validated** outputs
- **LLM-as-a-Judge** scoring with rubric-based evaluation and self-consistency
- **Automated safety and quality metrics** for reliable assessment
- **Reference-based text metrics** (ROUGE-L, BERTScore, embedding cosine, PPL)
- **Human evaluation UI** with blind A/B testing, Likert scales, and error tagging
- **Ray-powered parallel processing** for scalable batch inference
- **Provider-agnostic architecture** supporting Ollama (local), OpenAI (hosted), and vLLM

> **🔧 Current Default Configuration**  
> **Model Under Test:** `llama3:8b` via **Ollama** (local, macOS-friendly)  
> **Baseline & Judge:** **OpenAI** (`gpt-4o-mini` + `gpt-4o`)  
> 
> *All configurations can be modified via **`configs/model.yaml`***

---

## 🎯 What This Repository Does

The system evaluates wellness coaching LLMs by:

1. **Input Processing:** Takes daily health signals (RHR/HRV/sleep/steps + journal entries)
2. **Dual Model Testing:** Prompts both LLM and Baseline models with identical coaching tasks
3. **Schema Validation:** Enforces JSON outputs matching Pydantic schema (summary → suggestions → steps → disclaimer)
4. **Multi-Modal Scoring:** Evaluates results using:
   - **LLM judge** with multi-prompt self-consistency and weighted rubrics
   - **Automatic checks** for blocked content, disclaimer compliance, length, and safety
   - **Reference-based metrics** against silver standard references

All evaluation artifacts are saved to `out/` for longitudinal quality tracking and analysis.

---

## 🚀 Quick Start Guide

### Prerequisites
- **macOS** (or compatible system)
- **Python 3.11+**
- **Ollama** installed (`brew install ollama`)
- **OpenAI API key** for baseline and judge models

### Installation & Setup

```bash
# 1. Environment Setup
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -U "pyarrow==16.1.0" "ray[data]==2.20.0"

# 2. Local LLM Setup
ollama serve            # Keep running in separate terminal
ollama pull llama3:8b

# 3. OpenAI Configuration
echo 'OPENAI_API_KEY=sk-...YOUR_KEY...' >> .env

# 4. Data Preparation
python scripts/make_synthetic_data.py
python scripts/prepare_eval_splits.py
```

### Running Evaluations

```bash
# 5. Batch Inference with Schema Validation
python evals/runners/batch_infer.py

# 6. LLM Judge Scoring
python evals/runners/eval_llm_judge.py

# 7. Automatic Safety & Quality Checks
python evals/runners/eval_auto.py

# 8. Reference-Based Text Metrics
python scripts/refs_from_judge.py --limit 50
python evals/runners/eval_ref_metrics.py --skip-ppl --limit 50

# 9. Human Evaluation Interface
streamlit run human/annotator_app.py
python scripts/analyze_human_eval.py

# 10. Scalable Processing with Ray
python evals/runners/ray_eval.py \
  --config configs/model.yaml \
  --split evals/datasets/test.jsonl \
  --prompt coach_v1.jinja \
  --tag mut_v1 \
  --outdir out/ray/mut_v1 \
  --num-actors 3 --batch-size 4 --limit 20

python scripts/collect_ray_outputs.py out/ray/mut_v1 out/infer/mut_v1.ray.jsonl
mv out/infer/mut_v1.ray.jsonl out/infer/mut_v1.jsonl
python evals/runners/eval_llm_judge.py
python evals/runners/eval_auto.py
```

---

## 📊 Architecture Overview

### Pipeline Flow
![Pipeline Overview](https://github.com/mohandasnj/health-evals/blob/main/img/pipeline.png)

### Provider Abstraction
![Provider Architecture](https://github.com/mohandasnj/health-evals/blob/main/img/providers.png)

### End-to-End Execution
![Sequence Diagram](https://github.com/mohandasnj/health-evals/blob/main/img/sequence.png)

---

## 📁 Project Structure

### Configuration Files
- **`configs/model.yaml`** — Model and provider settings (local LLM, baseline, judge)
- **`configs/judge.yaml`** — Rubric dimensions, weights, and scoring parameters

### Core Applications
- **`apps/providers.py`** — Provider abstraction layer (Ollama/OpenAI/vLLM) with retry logic
- **`apps/wellness_coach/schemas.py`** — Pydantic v2 schema definitions for Guardrails
- **`apps/wellness_coach/prompt_templates/`**
  - **`coach_v1.jinja`** — JSON-only prompt template (local models)
  - **`coach_v2.jinja`** — Conversational variant (baseline models)

### Evaluation Runners
- **`evals/runners/batch_infer.py`** — Prompt rendering → provider calls → validation → `out/infer/*.jsonl`
- **`evals/runners/eval_llm_judge.py`** — Rubric scoring with self-consistency → `out/judged/*.jl`
- **`evals/runners/eval_auto.py`** — Deterministic safety/quality checks → `out/metrics/*.csv`
- **`evals/runners/eval_ref_metrics.py`** — Text similarity metrics → `out/metrics_ref/*.csv`
- **`evals/runners/ray_eval.py`** — Parallel inference using Ray

### Data & Utilities
- **`scripts/make_synthetic_data.py`** — Synthetic dataset generation
- **`scripts/prepare_eval_splits.py`** — Dataset splitting and preparation
- **`scripts/refs_from_judge.py`** — Silver reference generation via judge model
- **`scripts/collect_ray_outputs.py`** — Ray output aggregation
- **`scripts/analyze_human_eval.py`** — Human evaluation analysis
- **`evals/datasets/*.jsonl`** — Input datasets

### Human Evaluation
- **`human/annotator_app.py`** — Streamlit-based blind A/B rating interface → `out/human/annotations.csv`

---

## 📈 Results & Evaluation Metrics

Based on preliminary testing with 5 sample entries (expandable to ~300 total test entries), **GPT-4o-mini** shows superior performance compared to **Ollama**, though potential judge bias should be considered given the OpenAI judge model.

### 🏆 LLM Judge Scores

**File:** `out/judged/baseline_v2.jl` (JSON Lines format)

**Description:** Multi-prompt averaged ratings using weighted rubric scoring

**Key Fields:**
- **`id`** — Dataset example identifier (e.g., `ex0`)
- **`tag`** — System/run identifier (e.g., `baseline_v2`)
- **`blocked`** — Schema/JSON validation failure status
- **`dim_scores`** — Individual dimension scores (1-5):
  - **Helpfulness** — Practical utility of advice
  - **Factuality** — Accuracy of health information
  - **Safety** — Adherence to safety guidelines
  - **Clarity** — Communication effectiveness
- **`final`** — Weighted overall score (using `configs/judge.yaml` weights)

![Judge Evaluation Results](https://github.com/mohandasnj/health-evals/blob/main/img/judged.png)

### ⚡ Automatic Quality Metrics

**File:** `out/metrics/baseline_v2.csv` (CSV format)

**Description:** Fast deterministic quality indicators requiring no judge model

**Key Columns:**
- **`blocked`** — Schema reliability indicator
- **`len_chars`** — Output length (verbosity/truncation detection)
- **`has_disclaimer`** — Required disclaimer presence (0/1)
- **`safety_hits`** — Policy violation count (dosing/diagnosis/crisis patterns)

![Automatic Metrics](https://github.com/mohandasnj/health-evals/blob/main/img/met.png)

### 📝 Reference-Based Text Metrics

**File:** `out/metrics_ref/baseline_v2.csv` (CSV format)

**Description:** Similarity and fluency metrics against silver standard references

**Key Metrics:**
- **`rougeL_f`** — ROUGE-L F1 score (lexical overlap)
- **`bertscore_f1`** — Semantic similarity via BERTScore
- **`embed_cosine`** — Sentence embedding cosine similarity
- **`ppl_gpt2`** — GPT-2 perplexity (fluency indicator, lower = better)

![Reference-Based Metrics](https://github.com/mohandasnj/health-evals/blob/main/img/ref.png)

---

## 🖥️ Human Evaluation Interface

### Running the Interface
```bash
streamlit run human/annotator_app.py
```

### Features
- **Blind A/B Testing** — Choose dataset and model pairs with masked labels
- **Comprehensive Rating** — Side-by-side model outputs with structured JSON rendering
- **Multi-Dimensional Scoring** — Likert scales for each rubric dimension
- **Error Tagging** — Flag unsafe, non-actionable, or problematic content
- **Persistent Storage** — Ratings saved to `out/human/annotations.csv` with full provenance
- **Analysis Tools** — Use `scripts/analyze_human_eval.py` for agreement and winner rate analysis

### Interface Screenshots
![Streamlit Interface - Home](https://github.com/mohandasnj/health-evals/blob/main/img/streamlit1.png)
![Streamlit Interface - Rating](https://github.com/mohandasnj/health-evals/blob/main/img/streamlit2.png)

---

## 🔬 Implemented Evaluation Methods

### **LLM-as-a-Judge with Self-Consistency**
- **Implementation:** `evals/runners/eval_llm_judge.py`, `configs/judge.yaml`
- **Output:** `out/judged/*.jl`
- **Features:** Multi-prompt consensus, weighted rubric scoring

### **Automatic Metric-Based Evaluations**
- **Safety Heuristics:** `evals/runners/eval_auto.py` → `out/metrics/*.csv`
- **Text Quality Metrics:** `evals/runners/eval_ref_metrics.py` → `out/metrics_ref/*.csv`
  - ROUGE-L, BERTScore-F1, embedding cosine similarity, optional perplexity

### **Human Evaluation System**
- **Interface:** `human/annotator_app.py` (Streamlit-based)
- **Data Storage:** `out/human/annotations.csv`
- **Analysis:** `scripts/analyze_human_eval.py`

### **Scalability Infrastructure**
- **Parallel Processing:** `evals/runners/ray_eval.py`
- **Output Management:** `scripts/collect_ray_outputs.py`

### **Schema Enforcement & Guardrails**
- **Schema Definition:** `apps/wellness_coach/schemas.py` (Pydantic v2)
- **Prompt Engineering:** `apps/wellness_coach/prompt_templates/*.jinja`

---

## 🛣️ Roadmap & Future Development

### **Health-Grounded Datasets**
- Integrate consumer-safe subsets of **MultiMedQA** and **HealthSearchQA**
- Add **MedSafetyBench**-style safety categories
- Develop curated wellness-coach evaluation suite

### **Enhanced Human Annotation**
- Transform Streamlit app into professional rater console
- Implement **gold standard items** and **attention checks**
- Add **adjudication workflows** and **quality control reporting**
- Track **inter-rater reliability (IRR)** and **annotation drift**

### **Judge Model Improvements**
- Increase **self-consistency** validation rounds
- Develop **style-blind prompting** techniques
- Conduct **meta-evaluation** against human ratings for correlation analysis

### **Health Safety & Alignment**
- Deploy comprehensive **safety rubric** (diagnosis/dosing restrictions, escalation protocols)
- Implement **automatic policy filters**
- Build **red-team generators** with tracked safety metrics

### **Advanced Failure Analysis**
- **Failure case clustering** and classification
- **Adversarial test suite** maintenance for regression prevention
- **Root cause analysis** tooling for systematic improvements

---

## 📄 License & Contributing

This project is designed for research and educational purposes in AI safety and healthcare applications. Please ensure compliance with relevant healthcare regulations and ethical guidelines when using this evaluation framework.

For questions, issues, or contributions, please refer to the project's GitHub repository.
