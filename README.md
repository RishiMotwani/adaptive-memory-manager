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

### Option A — Docker (easiest, runs anywhere)
Requires [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2.

```bash
docker compose up -d
```

This brings up three services:

| Service          | Purpose                                                        |
|------------------|----------------------------------------------------------------|
| `ollama`         | Embedded Ollama server (port 11434) acting as the local LLM backend |
| `ollama-pull`    | One-shot boot job that pulls `llama3.1:8b` + `nomic-embed-text` on first run, then exits |
| `app`            | The FastAPI memory manager + web dashboard (port 9000)          |

Open the dashboard at **http://localhost:9000/dashboard**.

Notes:
- First run downloads the Ollama models (~5 GB) and can take a while; subsequent starts are fast because models persist in the `ollama-models` volume.
- The app waits for model pulls to finish (`service_completed_successfully`) before starting.
- If your machine does **not** have Ollama installed, this just works — everything is containerized.
- Ollama model files are stored in a named Docker volume (`ollama-models`), so they survive `docker compose down`. Remove the volume with `docker compose down -v` to force a clean model re-pull.

**Using an existing Ollama** (already running on your host, e.g. port 11434): the bundled Ollama service will conflict with it. Stop the host Ollama, or skip the bundled services and run just the app pointed at your host:

```bash
docker build -t adaptive-memory-app .
docker run -d -p 9000:9000 \
  --add-host host.docker.internal:host-gateway \
  -e OLLAMA_ENDPOINT=http://host.docker.internal:11434 \
  adaptive-memory-app
```

### Option B — From source
```bash
pip install -r requirements.txt
python server.py
```

Open **http://localhost:9000/dashboard**.

Requirements:
- Python 3.11+
- A running [Ollama](https://ollama.com) instance with `llama3.1:8b` and `nomic-embed-text` pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```

### Configuration
All settings live in `config.yaml` (decay lambdas, scoring weights, retrieval `top_k`, token budget, etc.). The following environment variables override relevant config values at runtime:

| Variable             | Default            | Description                                |
|----------------------|--------------------|--------------------------------------------|
| `OLLAMA_ENDPOINT`    | `http://localhost:11434` | Where Ollama is reachable            |
| `LLM_MODEL`          | `llama3.1:8b`      | LLM used for fact extraction/answering     |
| `EMBEDDING_MODEL`    | `nomic-embed-text` | Embedding model for retrieval               |

---

## The Dashboard (`/dashboard`)

The single-page dashboard is the live demo harness. It runs a real LLM, replays synthetic conversation scenarios, and pits the adaptive memory system against a "barebones" baseline (a raw sliding-window transcript) on the *same* turns.

### Top bar controls
- **Model selector** — pick which Ollama model runs extraction and answering. Changes are applied immediately.
- **Token budget** (`Apply` box) — the context window budget in tokens. The demo compares both systems against this cap. Default 4096.
- **VRAM chip** — sampled GPU VRAM usage, estimated KV-cache savings (Δ MiB), and the active model.
- **Turn count** — number of synthetic turns in the demo (20–500). Also available as a preset dropdown.
- **Re-mention density (%)** — what fraction of turns re-mention planted facts (simulates real-world repetition). Higher density = facts resurface more often.
- **Planted facts** — how many distinct facts the scenario plants into the conversation (1–50).
- **Filler** — routine-turn length in tokens (`short`/`medium`/`long`); longer filler puts more eviction pressure on the baseline.

### Actions
- **Load Demo** — starts a synthetic scenario with the given parameters. A progress bar shows per-turn extraction progress with an ETA, because every turn is parsed by the real Ollama model. Polling updates all panels live. *Cancel Demo* aborts mid-run.
- **Reset Memory** — cancels any running demo and purges all stored/state memory via the ethics purge endpoint, returning the dashboard to a clean slate.

### Dashboard panels
1. **Live Conversation** — the ongoing chat transcript. When retrieval finds relevant memories, their facts are shown as category tags inline. When a demo has answers, both the **Optimized** (curated-memory) model answer and the **Barebones** (raw-window) model answer appear side by side.
2. **Adaptive Memory State** — every memory currently held:
   - **Active Memories** — fact, category, originating turn, and current importance `M(t)`.
   - **Pruned / Forgotten** — deliberately forgotten memories and why they were pruned.
3. **Optimized vs Barebones — Live Comparison**:
   - Two token bars: the raw transcript's context usage vs the curated memory's injected tokens, both measured live against the budget.
   - **Barebones** side shows which turn numbers have already been dropped from the raw window, how many planted facts are still in the window vs already evicted, and how many of those the adaptive system still remembers.
   - **Optimized** side shows active/pruned counts, which retrieval engines ran (lexical/embedding), and a ⚠ warning when the token budget forced adaptive evictions (with the fact shown).
   - **"Barebones forgot these planted facts, but you still remember"** — a direct, per-fact list of what the adaptive system kept that the sliding window lost. Both-lost facts are flagged honestly.
4. **Context Growth per Turn** — an SVG chart plotting raw cumulative transcript tokens (red) vs adaptive memory tokens (green), with the budget drawn as a dashed line. Demonstrates that the optimized context stays flat while the raw window grows linearly.
5. **What the Barebones Model Actually Sees** — preview of the current raw-window turns (green) vs earlier evicted turns (red), with token counts.
6. **Both Sides' Memory — Same Facts, Different Fate** — two-column view of each system's remembered/forgotten memory, plus a planted-fact **head-to-head table** (✔ remembers / ✘ forgot) for the proposed vs barebones systems.
7. **Per-Turn Memory Log** — a live chronological feed with, for each turn: how many facts the optimized side injected, its injected-token count, budget-squeeze warnings, how many it pruned that turn, the baseline's window size, how many facts the baseline saw, and any facts the baseline newly lost that turn.
8. **Live Experiments (E1–E6)** — one-click statistical evaluation of the *currently loaded scenario*, so you can iterate on parameters and instantly re-measure.

### Detail views (top subnav)
- **Dashboard** — everything above.
- **Barebones Window** — the complete raw context window turn-by-turn (what the baseline model actually receives) and the full evicted list, paged.
- **Memory** — full tables of active/pruned memories for both systems plus the complete planted-fact head-to-head, paged with **Load more**.
- **Timeline** — the full per-turn memory log with Prev/Next paging across the whole scenario.
- **Experiments** — full E1–E6 results with a toggle to inspect the raw manifest JSON.

### The experiments (E1–E6)
Run via **Run Experiments (E1–E6)** on the dashboard or the Experiments view:

- **E1 — Token Efficiency:** paired **Wilcoxon signed-rank** test on per-turn injected context tokens, proposed vs baseline, with p-value and significance marker.
- **E2 — Memory Accuracy:** positive recall (were planted facts found?) and forgetting precision (were transient facts correctly dropped?) for both systems, computed against ground truth (real LLM replay when available, estimate otherwise).
- **E3 — Pipeline Latency:** breakdown per stage — extraction, window replay, answer, scoring, compression, decay, budget evict, retrieval, embedding.
- **E4 — Needle-in-Haystack:** recall accuracy binned by recency distance (≥1, ≥5, ≥10, … turns back) for proposed vs baseline. Shows how far back the adaptive system can still recall facts the baseline has already evicted.
- **E5 — Ablations:** replay of the same scenario with one component disabled each (e.g., no decay, no importance scoring, no embeddings) — accuracy, token overhead vs the full system, and pruned count per ablation.
- **E6 — Statistical Power:** observed Cohen's *d* for E1, the required sample size per group at power 0.80 and α 0.05, and the turns that were actually evaluated.

### API endpoints (used by the dashboard)
| Method | Path                          | Purpose                                        |
|--------|-------------------------------|------------------------------------------------|
| POST   | `/api/chat`                   | Send a turn; returns answer, retrieved context, memory diff, comparison |
| GET    | `/api/models`                 | List reachable Ollama models + active one      |
| POST   | `/api/models/select`          | Change the active LLM model                    |
| GET/POST | `/api/settings`             | Read or update `max_context_tokens` (and other settings) |
| POST   | `/api/demo/scenario`          | Start a synthetic scenario replay              |
| GET    | `/api/demo/job`               | Poll demo progress                             |
| POST   | `/api/demo/job/cancel`        | Cancel a running demo                          |
| GET    | `/api/state`                  | Current memory state (active/pruned, comparison) |
| GET    | `/api/timeline`               | Recent per-turn memory log                     |
| GET    | `/api/timeline/full`          | Paged full timeline                            |
| GET    | `/api/comparison/full`        | Full baseline window + both-side memory detail |
| GET    | `/api/experiments/latest`     | Latest E1–E6 results                           |
| POST   | `/api/experiments/run`        | Run E1–E6 on the loaded scenario               |
| GET    | `/api/gpu`                    | GPU VRAM / KV-cache sampling                   |
| GET    | `/api/ethics/privacy_export`  | Export stored data (privacy control)           |
| DELETE | `/api/ethics/purge`           | Wipe all stored memory (privacy control)       |
| GET    | `/dashboard`                  | The web UI                                     |