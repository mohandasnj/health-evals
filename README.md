# Health Evals — LLM Evaluation for Wellness Coaching

End-to-end evaluation project for wellness coaching.

- ✅ A/B comparisons (Model-Under-Test vs Baseline)
- ✅ Guardrails + Pydantic schema enforcement
- ✅ LLM-as-a-Judge (rubric + self-consistency)
- ✅ Automatic safety/quality checks
- ✅ Ray for scalable, parallel inference
- ✅ Pluggable providers (Ollama, OpenAI, vLLM) via a single YAML config

---

## Prerequisites (macOS)

- **Python 3.11+**
- **Homebrew**
- **Ollama** (local model runtime)
- **OpenAI** account with **billing enabled** (for baseline/judge)

Install & start Ollama:
```bash
brew install ollama
ollama serve
ollama pull llama3:8b

# To run: 

# from your repo root, venv active
# python scripts/make_synthetic_data.py
# python scripts/prepare_eval_splits.py

# python evals/runners/batch_infer.py     # MUT -> Ollama (or any other model if you changed it)
# python evals/runners/eval_llm_judge.py  # judge -> OpenAI (or any other model if you changed it)
# python evals/runners/eval_auto.py
