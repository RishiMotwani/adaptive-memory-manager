# Adaptive Memory Manager for Context-Constrained LLMs

> **PRIMARY CLAIM:** A category-conditioned exponential decay function, combined with importance-weighted retrieval, improves long-horizon recall accuracy per token compared to static memory-scoring methods (RAG, sliding window, summarization-only) — and does so by forgetting the right things, not just remembering more.

## Architecture & Paper Scope
This repository implements a category-conditioned decay memory optimization architecture targeting local SLMs (e.g., Llama 3.1 8B, Qwen 2.5 7B, Phi-3-mini) running under strict context constraints (4k/8k windows).

### Key Features
1. **Category-Conditioned Decay Engine:** Grounded in Ebbinghaus & SuperMemo SM-2 forgetting curves.
2. **Multi-Factor Scoring & Ablation Engine:** Linear and logistic weight fitting across relevance, utility, recency, and frequency.
3. **Rigorous Experimental Suite (E1–E6):** Paired Wilcoxon signed-rank tests, statistical power analysis, and needle-in-a-haystack visualizations.
4. **Privacy & Ethics Compliance:** Native zero-knowledge local storage, strict forgetfulness guarantees, and programmatic export/purge controls.

## Quick Start
```bash
pip install -r requirements.txt
python run_experiments.py
python server.py
```
