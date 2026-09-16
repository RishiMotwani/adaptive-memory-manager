# Adaptive Memory Manager — Research Brain

Living document. Every file edit after a research session should trace back to a
numbered design decision (D1, D2 …) here, with the citation that motivated it.
Nothing goes into `memory_optimizer/` or `server.py` without first landing here.

## 0. The one-sentence takeaway

Across ~15 papers, the single most consistent empirical result is:

> **Retrieval quality matters far more than how you write facts.** The gap
> between a weak and strong retriever (14–23 accuracy points) dwarfs the gap
> between a good writer (LLM fact extraction) and a naive writer (raw chunks,
> 3–8 points). Constructed/extracted memory is *not* inherently better than
> direct retrieval over raw turns — and contextual compression ("summarizing
> memory") often *hurts* by silently drifting.

Our current system inverts this priority: it spends all its complexity on
extraction/scoring/deduping and none on retrieval (lexical `_token_overlap` in
`memory_optimizer/retrieval.py:5`). The highest-leverage change is therefore the
shift from lexical to real embedding retrieval — exactly the follow-up we
planned.

---

## 1. Big picture / framework papers

### 1.1 MemGPT — "LLMs as Operating Systems"
- Packer et al., "MemGPT: Towards LLMs as Operating Systems", arXiv:2310.08560. (search: par.nsf.gov/servlets/purl/10524107)
- Virtual context management as OS paging: a **main context** window plus
  external memory tiers (**recall database** for recent conversation, **archival
  vector store** for facts), plus a **search function** the model itself calls.
- Model issues *function calls* to its own memory manager: `memory_insert`,
  `memory_search`, `cursor_search`, etc. Governance of what to save is explicit
  "self-editing" by the agent, not implicit.
- Eviction: FIFO queue + a **warning token count** — the model is told "you are
  approaching context limit, must evict" and chooses what to drop.
- Perf: beats fixed-context on document analysis + multi-session chat; ~10× cost
  savings; the *utility* (roughly our adaptive-vs-baseline measurement) was
  measured both subjectively and via DMR (Deep Memory Retrieval) benchmark.
- **For us**: our `top_k:5` fixed injection is MemGPT's shallow end. The lesson
  is that memory management should be *adaptive to context pressure*. Our
  min(20,500)-turn demo + 4096 `max_context_tokens` is already a MemGPT-style
  paging arena; we lack the "warning count / evict-under-pressure" mechanic on
  the adaptive path (see D1/D2).

### 1.2 "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, Emerging Frontiers" — arXiv:2603.07670
- Organizing framework: **write → manage → read** loop, with memory taxonomy by
  temporal scope, representational substrate, control policy.
- Five mechanism families: context-resident compression; retrieval-augmented
  stores; reflective self-improvement; hierarchical virtual context (MemGPT);
  policy-learned management (RL).
- Repeated finding: **"gap between has-memory / no-memory > gap between
  backbones"** — memory is the dominant driver of agent quality, larger than the
  model itself.
- Key pathologies:
  - **Summarization drift**: context-resident compression (rolling summaries)
    silently discards low-frequency facts; errors compound turn over turn.
    RET-LLM & A-Mem respond with extractive atomic facts / triplets instead.
  - Debugging difficulty: when a memory system fails, you cannot tell whether
    the write path, retrieval, compression, or down-stream reasoning caused it.
    *Our `metric_source` flags + live-real-token E-values are a partial answer
    (we can attribute failures to E1 retention vs E3 cost vs E5 reuse).*
- Retrieved memories mix **recency (exponential decay) + relevance**; solid
  baseline, matches our decay formulation.

### 1.3 Cognitive-architecture framing (LightMem, Sumers et al. survey)
- LightMem (ICLR 2026; zjunlp/LightMem) is the Atkinson–Shiffrin model made
  concrete: **sensory** (cheap fast filter) → **short-term** (topic grouping +
  consolidation) → **long-term** (offline "sleep-time" consolidation that decouples
  the expensive writes from the online path).
- Reports accuracy gains up to 10.9% on LongMemEval while cutting token *usage*
  up to 117× and API calls up to 159× (they "summarize once, look up sorted" so
  recall is cheap) — but see 1.7: its substantive win is contested.
- **For us**: our ingest-time extraction + dedupe is the "online" write, and we
  have no offline consolidation pass. A cheap consolidation target = the decay
  prune + re-embed pass that happens on a *replay* timer, not inline.

---

## 2. Priority finding: retrieval > write, and constructed memory ≠ better

### 2.1 "Diagnosing the Retrieval vs. Utilization Bottleneck in LLM-based Generative Memory Systems" — arXiv:2603.02473
- Ablation over write strategy × retrieval method × downstream reasoning.
- **Retrieval method drives 14–23 accuracy points; write strategy contributes
  3–8 points.**
- **Basic RAG over raw chunks ≥ Mem0-style extraction and ≥ MemGPT rolling
  summaries** across retrieval methods. Simpler "writer" loses little or
  nothing.
- Cosine / BM25 / hybrid distinction matters more than whether your write step
  used an LLM.
- **For us**: our entire ingest pipeline (LLM extraction, scoring, dedupe,
  keyword categories) is optimizing the *low-leverage* half. Budget real effort:
  swap `_token_overlap` for real embeddings first.

### 2.2 "Reproducing LightMem: Naive RAG Is Just as Good for Memory Management" — arXiv:2607.29104
- Matched retrieval depths, token budgets, oracle eval. Result: **constructed
  memories (LightMem MemBank entries) don't beat direct retrieval over raw
  dialogue turns** ("Naive RAG").
- Retriever choice is huge: same store, Recall@10 = 0.390 (BM25) vs 0.587
  (Qwen3-Embedding-4B); answer accuracy 58.1% → 75.5%.
- LightMem = a *context-efficiency* trade, not an accuracy win.
- **For us**: our `baseline_replay` (raw window) vs `adaptive_replay` (fact
  store) is exactly this contest, and in our one real run adaptive won on
  accuracy at ~71 vs ~496 tokens (D1 baseline) — but our retrieval is the weak
  BM25-equivalent, so we have headroom to keep the win *and* the small context.

### 2.3 Agent memory reranking evidence (MemReranker — arXiv:2605.06132; Zep — arXiv:2501.13956)
- MemReranker: retrieval quality is "the bottleneck of agent memory"; a small
  (0.6B) *reasoning-aware reranker* matches GPT-4o-mini / Gemini-3-Flash on
  LoCoMo at ~200ms. Reranking = second pass over vector candidates.
- Zep: hybrid recall via cosine + BM25 + graph BFS, then rerank (RRF, MMR,
  cross-encoder, and a graph "episode-mentions" reranker that boosts whatever
  was *frequently mentioned* — i.e., reinforcement from use).
- **For us**: `nomic-embed-text` is already installed in Ollama (verified
  `GET /api/tags`). Target retrieval pipeline = embedding cosine candidate
  gen → rerank within candidates by our existing importance blend. Zep's
  frequency-boost validates our "re-mention reinforcement" idea (D4).

---

## 3. Compression and summarization: mostly a trap

### 3.1 Summarization drift (survey 2603.07670; MemGPT companion findings)
- Rolling/contextual summarization loses low-frequency facts and compounds
  error. RET-LLM, A-Mem, Mem0 all choose **atomic self-contained facts +
  embeddings** over summaries.
- **For us**: our `compress_cluster` (memory_optimizer/compression.py:16) is a
  *fake* summarizer — it concatenates facts under a "Summarized Context:"
  prefix and never calls a model. Research says: (a) don't pretend to summarize,
  (b) real LLM summarization is the highest-risk maneuver in the whole memory
  stack, (c) the honest alternative is harder dedupe + entity linkage, not
  compression. **Recommendation: keep compress_cluster concatenative at most,
  or better, make dedupe the consolidation mechanism and delete the
  Summarized-Context block** (see D6). A "real" model summarizer is a
  *risky* feature, not a safe upgrade, on 8GB VRAM.

### 3.2 Human-inspired alternatives to summarization
- **Human-Inspired Memory Models** (arXiv:2605.08538): sleep-phase
  consolidation, interference-based forgetting, **engram maturation** (new
  memories start as latent "silent" traces at activation_strength 0.0 and only
  become recallable after repeated reinforcing retrieval) — a different, nicer
  model than our one-shot `base_score`. Reconsolidation: *retrieval rewrites the
  memory*. Entity knowledge graphs + hybrid multi-cue retrieval (episodic vector
  + semantic KG).
- **NEMORI / "What deserves memory"** (ACL 2026): distil on **prediction error**
  — retain what the agent *fails to predict* from existing knowledge; "what is
  predictable is redundant." Training-free. Critique: heuristic importance tags,
  emotional tags, rigid factual templates are all wrong signals.
- **Mem0** (Chhikara et al., "VanillaRAG/RAG2"): facts extracted as
  self-contained, embedded, and **conflict-resolved (add / update / noop)** —
  a memory write is a semantic *merge*, not an append. We have token-overlap
  dedupe but not semantic conflict resolution (full dedupe happens only within
  turn; `dedupe_incremental` compares to the store but with lexical overlap).
- **For us** (D4-D6): replace one-shot-save with: engram-strength activation on
  re-mention (memory is born latent, grows with reinforcement, decays without);
  count retrieval-reinforcement (Zep episode mentions + Oblivion); treat dedupe
  as the compression story. Our `access_count` field exists but nothing uses it
  for score — that's the hook.

---

## 4. Forgetting: static decay is the weakest part

### 4.1 SF-AMS — "Strategic Forgetting for Structured Memory in LLM Agent" — arXiv:2607.22562
- Replaces static retrieval + **heuristic decay** with a **utility-driven
  survival** mechanism: the importance of a memory is *updated online from usage
  redundancy + temporal access signals*, not assigned once at write.
- Composite Importance Scoring (semantic + entity). Outperforms LightMem, Mem0,
  A-Mem on LoCoMo / LongMemEval-s; largest gain on multi-hop under
  Qwen2.5-7B (+9.65 F1).
- **For us**: our decay (memory_optimizer/decay.py: `M(t) = base_score·e^(−λΔ)`)
  is a *static* curve, exactly the thing SF-AMS replaces. `access_count` is
  already stored — wire it into importance so re-mentions slow decay and
  one-shot facts age out fast (D4/D5). This is the strongest research backing we
  have for a change to scoring/decay.

### 4.2 Oblivion — EMNLP 2026
- **Decay-driven activation**, not deletion: memories decay along the Ebbinghaus
  forgetting curve by `n_turns_since_last_access`; every *access* reinforces.
- Utility proxy `U_t`, access-frequency proxy `F_t`, decay temperature `T`
  (comfortably configurable). Read path uses *uncertainty* to decide whether to
  query memory at all → 73% token-cost cut at 120K.
- **For us**: validate that discounting should key on **time-since-last-access**
  (reinforcement resets the clock) rather than time-since-creation. Also: an
  "uncertainty-gated retrieval" would let the adaptive path skip injection on
  easy/quasi-redundant turns → direct token savings. Nice E3 lever (D7).

### 4.3 MemoryBank (Zhong et al., AAAI2024) & ACT-R vector learners
- MemoryBank: Ebbinghaus-curve decay for chatbot memory, self-consistent memory
  updates — the original "fade unless reviewed" design.
- ACT-R-style recalling (button-maze / generalisation study): activation =
  **temporal decay + semantic similarity + noise**; retrieval is biased by both
  recency *and* associativity, with probabilistic noise. "Forgetting as a
  feature" — old, unused memories are cheap to recover *if* you keep a
  low-cost trace (embedding) and let the value decay.

### 4.4 RecMem — ACL 2026 Findings
- **Subconscious memory layer**: stores lightweight embeddings only; only invokes
  the LLM to *consolidate* when it observes **sustained recurrence**. Lazy
  consolidation recovers facts plain extraction omits, at large token savings.
- **For us (D8)**: don't run expensive LLM extraction every single turn. Two-tier
  write: cheap embedding-index on every turn; LLM fact-extraction + consolidation
  only on recurrence (or deterministically, the demo's planted turns). Cuts
  extraction tokens + latency and mirrors our E3 numbers (extraction 513ms/turn
  is our dominant cost).

### 4.5 "How Memory Management Impacts Agents" (hit from search #3)
- Experience-following behavior correlates with manager decision quality;
  memory *addition* and *deletion* both measurably change an agent's behavior on
  future tasks — use future-task performance as *free quality labels* for the
  manager. Matches our idea of treating E1 retention-vs-used as a signal.

---

## 5. Context / representation mechanics

### 5.1 "Choosing How to Remember" — arXiv:2602.14038 (STIM/MTEM/LTSM)
- Three-tier sensorimotor→intermediate→long-term store, retrieval+fuse on the
  read path. Fine-grained tiering yields better trace separation than one flat
  store.

### 5.2 "Structural Memory of LLM Agents" — arXiv:2412.15266
- Retrieval methods compared: single-step, iterative (query-refining, RepoCoder
  style), multi-hop rerank. Embeddings matter; structure of memory affects
  results across 4 tasks.

### 5.3 "Active context compression" / Focus — arXiv:2601.07190
- Sawtooth pattern: **consolidate → withdraw** from context. Once useful
  content is confirmed, summarize into a knowledge block and *delete the raw
  logs*. StreamingLLM / LLMLingua as sub-components.
- **For us**: legalizes our `baseline_replay` window (raw recent turns) + a
  separate injected fact set: the baseline is the fresh context, the adaptive
  block is the consolidated knowledge. Cleanest framing for the demo narrative.

### 5.4 'Active context compression' trade-off we already measure
- E3 shows window_replay 1140ms vs adaptive answer 228ms — both real.
  Compression in the literature is about *prompt* cost; we measure both token and
  ms. Keep `debug.timing` per stage so history is auditable (already stored as
  fact_token_measurement, extraction ms, replay ms, answer ms).

---

## 6. Coding-agent-specific memory

### 6.1 RepoCoder (Zhang et al., EMNLP2023 — arXiv:2303.12570)
- Repo-level completion via **iterative retrieval–generation**: generated output
  itself rounds-trips as the retrieval query → converge to the relevant code.
  Two real takeaways: (1) relevance query = last observation, not the whole
  history; (2) recursion beats one-shot top-k for code.
- **For us**: our adaptive replay currently uses the *last turn* as the query
  (server `_adaptive_replay`), which matches "query = recent observation."
  Unevaluated: one refinement loop to re-query with the focused question.

### 6.2 CODEMEM — ACL2026 Findings ("AST-Guided Adaptive Memory for Repository-…")
- Memory manager for iterative repo-level code gen; guided by AST structure
  instead of raw text. Structural indexing (code graph / identifiers) > flat
  fact list for code contexts.
- **For us**: only if we ever ingest real code; our demo is conversation-shaped.
  Note the *principle*: index structure, not prose. Our `category` field is the
  proto-structure.

### 6.3 Evaluating AGENTS.md — arXiv:2602.11988 ("Are repository-level context files helpful?")
- **Context files don't generally improve success rates** and raise inference
  cost >20%; repository overviews are the least helpful section; instructions
  are followed but don't help. *Any* static context injection must be evaluated
  before keeping it.
- **For us**: the demo's "planted facts → re-mentioned" makes prefetchable static
  context and lets us *measure* whether injected context actually helps recall —
  exactly this paper's demanded discipline. Keep the comparison honest: report
  turns where injected facts changed the answer vs merely added tokens (E2
  exact-match machinery).

### 6.4 AceCoder (arXiv:2303.17780) / CODEAGENT (ACL2024)
- AceCoder: guided code generation + **example retrieval** (similar problem →
  few-shot is memory used as prompt). CODEAGENT: 5 integrated tools + tool-use
  strategies for repo-level tasks. Both: retrieval skills ≥ more model compute.

---

## 7. The five numbered design decisions this doc exists to justify

All trace to citations above. Decisions are written as "D# : current → target".
Pending items are marked ⏳ (need a human answer); recommended targets marked ★.

### D1 ⏳★ Shared token budget on the adaptive store
- *Evidence*: MemGPT warning-count eviction (§1.1); AGENTS.md cost discipline
  (§6.3); our first real run's 71-vs-496 comparative claim only stays
  comparable if both paths are bounded.
- *Current*: `baseline_replay` enforces `max_context_tokens:4096` on raw turns;
  adaptive store is unbounded (only dedupe + decay-prune + `top_k:5` cap).
  Optimized bar in UI saturates at 100% (`Math.min`), losing the comparison.
- *Target*: new `memory_optimizer/budget.py` with `token_budget_evict(memories,
  budget, fact_tokens)` shared by live pipeline and E5 replay; eviction policy
  **lowest `current_importance`, oldest tie-break** ★ (SF-AMS utility-driven
  survival, §4.1; MemoryBank fade, §4.3). Live injector also stays within budget
  using measured per-fact tokens (never overshoot).
- *Open question to user*: eviction order (a) lowest-importance/oldest ★,
  (b) oldest-first, (c) largest-token-first.

### D2 ⏳★ Explicit evict-under-pressure alerting (MemGPT "warning token count")
- *Target*: when the computed injection would exceed the budget, tag the
  comparison `budget_squeezed: true` and show *what had to be dropped* — turns
  lost-context counting into an honest metric. This converts the saturated-bar
  bug into a first-class signal.

### D3 ★ Real embedding retrieval (the highest-leverage fix)
- *Evidence*: §2.1 (14–23 pts from retrieval), §2.2 (0.390→0.587 Recall@10),
  §2.3 rerankers; §5.2 embeddings-as-structure.
- *Current*: `_token_overlap` lexical Jaccard only;
  `MemoryRetriever.__docstring__` claims "vector embedding cosine similarity"
  that doesn't exist (retrieval.py:15).
- *Target*: embed each fact ONCE at ingest via `nomic-embed-text` (already in
  Ollama) into a `fact_embedding` field; retrieve by cosine over candidates;
  rerank by `0.6·importance_blend + 0.4·similarity` (calibrate). Keep lexical as
  a *fallback* and record `retrieval_engine: embedding|lexical` so E5 stays
  comparable (`metric_source` precedent). `nomic-embed-text` gives 768-dim,
  ~0-dim token cost per embedding prompt — measure real latency, it's cheap.
  **Store embedding only as the trace (RecMem, §4.4); do NOT re-extract every
  turn** — see D8.

### D4 ⏳★ Re-mention reinforcement (engram activation + usage-fed importance)
- *Evidence*: SF-AMS dynamic importance (§4.1), Oblivion access-reinforcement
  (§4.2), Zep episode-mentions rerank (§2.3), Human-Inspired engram maturation
  (§3.2).
- *Current*: `access_count` recorded but unused by scoring; decay uses
  `source_turn_id`+`base_score` once, never refreshed; re-mentions do nothing.
- *Target*: on any retrieval/re-mention, **refresh the decay clock** and grow
  `current_importance` (usage-redundancy term). Budget-fed eviction (D1) then
  drops un-mentioned stale facts before reinforcement — matching
  MemoryBank/Oblivion forgetting curves. Fine-grained `injection_token_limit`
  ties in: injections are what *count* as reinforcement.
- *Open question*: exact reinforcement semantics — (a) refresh + recompute
  base_score from current `confidence`/relevance (★ recommended), (b) pure clock
  refresh only (Oblivion-style), (c) capped signal (prevent runaway).

### D5 ★ Dedupe as the compression story; kill the fake summarizer
- *Evidence*: summarization drift (§3.1); Human-Inspired "deduplicate, don't
  summarize" (§3.2); Mem0 semantic add/update/noop conflict resolution (§3.2).
- *Current*: `compress_cluster` concatenates under "Summarized Context:"
  (compression.py:23) — a mislabeled non-summarizer that can even *grow*
  token count; dedupe is lexical within-category only.
- *Target*: (a) `dedupe_incremental` upgraded to semantic merge using fact
  embeddings (D3) — conflict-resolve add/update like Mem0; (b) `compress_cluster`
  either removed from the live path or relabeled `concatenate_cluster` and never
  called during ingest; a *true* LLM summarizer is explicitly a later,
  opt-in, high-risk experiment.

### D6 ★ Category reintrospection / multi-cue retrieval
- *Current*: LLM category is overridden by a keyword sniff; category is a
  flat string used only by dedupe gating.
- *Target* (light): use category as an *entity/index tag* (Zep KG-lite,
  §2.3) in embedding + rerank; enables "topic fan-out" retrieval
  (multi-query per §5.2). Do NOT build a real KG yet.

### D7 ⏳★ Uncertainty-gated injection (read-path economy, Oblivion)
- *Evidence*: Oblivion §4.2 (73% token cut by not always querying);
  AGENTS.md §6.3 (needless context is a cost, not a benefit).
- *Target*: when top-1 embedding similarity is below an *adaptive* threshold,
  skip injection for that turn (still report `injected: "skipped-low-rel"` in
  the comparison). Direct E3/E5 win; must be toggleable for scenario parity.
  Threshold should auto-tune per model from calibration pass, like our
  `_calibrated_constants`.

### D8 ★ Lazy two-tier write (RecMem, LightMem sleep consolidation)
- *Evidence*: RecMem §4.4 (consolidate only on sustained recurrence);
  LightMem §1.3 (decouple expensive consolidation from online path).
- *Target*: cheap embed-on-every-turn (D3); full LLM extraction consolidated
  *only* when a fact recurs or (demo mode) deterministically on planted/filler
  schedule. Extraction is our dominant ms cost (513ms/turn) — this is the E3
  cost lever. Keep `extraction_fallback` counter so we can verify no quality
  loss versus full-extraction runs.

---

## 8. Open questions for the human (blocking decisions)

1. **D1 eviction policy**: lowest-`current_importance` + oldest tiebreak ★ / oldest-first / largest-token-first.
2. **D4 reinforcement semantics**: refresh + recompute base_score ★ / clock-refresh only / capped signal.
3. **D4/D1 demo controls**: `planted_count` cap — 50 leaving ≥20 filler turns ★ vs up to `turns/2`; and the exact meaning of the density control (re-mention rate of planted facts — user's 10-Sep mental model — vs "% of turns that carry a planted fact" which the current UI tooltip claims but the code ignores).
4. **D7 fill-mode**: when relevance candidates are scarce, fill the remaining `injection_token_limit` with best remaining by rank ★ vs strict top-k only (conservative, avoids injecting irrelevant context à la §6.3).

## 9. Benchmark / source index (how we'll verify changes)

- Our own: E1 (retention, real measured tokens), E2 (exact-match answer recall),
  E3 (latency+token cost, real prompt_eval_count), E5 (measured vs fallback
  token source, `token_source: measured`).
- External papers referencing these benchmarks we should read before any model
  swap: LoCoMo (MemReranker, SF-AMS use it), LongMemEval (LightMem, Zep, SF-AMS;
  LightMem sees "up to 10.9% gains"), Deep Memory Recall DMR (MemGPT, Zep).
  **Our demo is not a benchmark substitute** — it's an honest, real-everything
  single-conversation rig that tests comparative behavior under bounded context.
- Calibration constants (`_calibrated_constants`), per-fact token cache
  (`_fact_tokens`) and `metric_source` flags are the instrumentation that keeps
  every one of these numbers comparable across changes.