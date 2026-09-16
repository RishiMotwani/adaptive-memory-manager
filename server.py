import json
import os
import random
import subprocess
import threading
import time
import yaml
from difflib import SequenceMatcher
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests

from memory_optimizer.extraction import FactExtractor, build_extraction_prompt
from memory_optimizer.scoring import ImportanceScorer
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.retrieval import MemoryRetriever, _token_overlap
from memory_optimizer.compression import MemoryCompressor
from memory_optimizer.budget import token_budget_evict
from memory_optimizer.embeddings import embed_ollama

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Adaptive Memory Manager API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(BASE_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

if os.environ.get("OLLAMA_ENDPOINT"):
    cfg["system"]["ollama_endpoint"] = os.environ["OLLAMA_ENDPOINT"]
if os.environ.get("LLM_MODEL"):
    cfg["system"]["llm_model"] = os.environ["LLM_MODEL"]
if os.environ.get("EMBEDDING_MODEL"):
    cfg["system"]["embedding_model"] = os.environ["EMBEDDING_MODEL"]

extractor = FactExtractor(endpoint=cfg["system"]["ollama_endpoint"], model=cfg["system"]["llm_model"])
scorer = ImportanceScorer(weights=cfg["scoring_weights"])
decay_engine = CategoryDecayEngine(lambdas=cfg["decay_lambdas"], pruning_threshold=cfg["pruning"]["threshold"])
compressor = MemoryCompressor()
ollama_base = cfg["system"]["ollama_endpoint"]
embed_model = cfg["system"].get("embedding_model", "nomic-embed-text")


def _embed_texts(texts):
    return embed_ollama(texts, model=embed_model, endpoint=ollama_base)


retriever = MemoryRetriever(
    top_k=cfg["retrieval"]["top_k"],
    sim_threshold=cfg["retrieval"]["similarity_threshold"],
    embed_fn=_embed_texts,
    embedding_model=embed_model,
)

settings = {
    "max_context_tokens": int(cfg["system"]["max_context_tokens"]),
    "active_model": cfg["system"]["llm_model"],
    "top_k": int(cfg["retrieval"]["top_k"]),
    "similarity_threshold": float(cfg["retrieval"]["similarity_threshold"]),
    "pruning_threshold": float(cfg["pruning"]["threshold"]),
    "enable_compression": bool(cfg["compression"].get("enabled", True)),
    "injection_token_limit": int(cfg["system"].get("injection_token_limit", 0)),
}

state = {
    "current_turn": 0,
    "active_memories": [],
    "pruned_memories": [],
    "raw_history": [],
    "active_token_trace": [],
    "inject_token_trace": [],
    "ground_truth": [],
    "latency_stats": {},
    "timeline": [],
    "last_evicted_ids": [],
    "last_demo": {},
    "demo_job": {},
"baseline_replay": [],
    "last_context": {},
    "last_budget_evictions": [],
}

_gpu_cache = {"ts": 0, "data": None}

_token_constants = {"model": None}
_fact_token_cache = {}


class ChatMessage(BaseModel):
    user_message: str


def _measure_prompt_tokens(prompt: str, model: str = None, timeout: int = 60):
    model = model or settings["active_model"]
    try:
        resp = requests.post(f"{ollama_base}/api/generate", json={
            "model": model,
            "prompt": str(prompt),
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_predict": 1, "num_ctx": 8192},
        }, timeout=timeout)
        return resp.json().get("prompt_eval_count")
    except Exception:
        return None


def _calibrated_constants():
    model = settings["active_model"]
    if _token_constants.get("model") == model and all(
            _token_constants.get(k) is not None for k in ("C_extract", "C_replay", "C_answer")):
        return _token_constants
    c_extract = _measure_prompt_tokens(build_extraction_prompt("X"))
    c_extract = (c_extract - 1) if c_extract is not None else None
    replay_const = (f"{WINDOW_REPLAY_PROMPT}\n\n--- BEGIN TRANSCRIPT ---\n\n--- END TRANSCRIPT ---\n\nReturn JSON:")
    c_replay = _measure_prompt_tokens(replay_const)
    answer_const = (f"{ANSWER_PROMPT}\n\n--- CONTEXT ---\n(no context entries)\n---\n\nQuestion: X\nAnswer:")
    c_answer = _measure_prompt_tokens(answer_const)
    c_answer = (c_answer - 1) if c_answer is not None else None
    _token_constants.update(
        model=model, C_extract=c_extract, C_replay=c_replay, C_answer=c_answer,
        measured=c_extract is not None and c_replay is not None and c_answer is not None,
    )
    return _token_constants


def _extraction_content_tokens(meta: dict) -> int:
    constants = _calibrated_constants()
    count = meta.get("prompt_eval_count")
    c = constants.get("C_extract")
    if count is None or c is None:
        return None
    return max(0, count - c)


def _fact_tokens(fact_text: str, model: str = None) -> int:
    model = model or settings["active_model"]
    key = (model, fact_text)
    cached = _fact_token_cache.get(key)
    if cached is not None:
        return cached
    count = _measure_prompt_tokens(fact_text, model=model)
    if count is None:
        return None
    _fact_token_cache[key] = count
    return count


def _lookup_or_measure(text: str) -> int:
    return _fact_tokens(text)


def _persist_settings():
    with open(BASE_DIR / "config.yaml", "r") as f:
        doc = yaml.safe_load(f)
    doc["system"]["max_context_tokens"] = int(settings["max_context_tokens"])
    doc["system"]["llm_model"] = settings["active_model"]
    doc["retrieval"]["top_k"] = int(settings["top_k"])
    doc["retrieval"]["similarity_threshold"] = float(settings["similarity_threshold"])
    doc["pruning"]["threshold"] = float(settings["pruning_threshold"])
    doc["compression"]["enabled"] = bool(settings["enable_compression"])
    doc["system"]["injection_token_limit"] = int(settings["injection_token_limit"])
    with open(BASE_DIR / "config.yaml", "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _gpu_snapshot():
    now = time.time()
    if now - _gpu_cache["ts"] < 2 and _gpu_cache["data"] is not None:
        return _gpu_cache["data"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) == 2:
            _gpu_cache.update(ts=now, data={"used_mib": int(parts[0]), "total_mib": int(parts[1])})
    except Exception:
        _gpu_cache.update(ts=now, data=None)
    return _gpu_cache["data"]


def _model_family(model: str) -> str:
    m = model.lower()
    for fam in ("llama", "qwen", "gemma", "mistral", "deepseek", "moondream", "phi"):
        if fam in m:
            return fam
    return "default"


def _kv_bytes_per_token() -> int:
    table = cfg["vram"]["bytes_per_token_estimate"]
    return int(table.get(_model_family(settings["active_model"]), table["default"]))


def _vram_block(baseline_tokens: int, optimized_tokens: int) -> dict:
    sample = _gpu_snapshot()
    bpt = _kv_bytes_per_token()
    est_b = baseline_tokens * bpt / 1048576
    est_o = optimized_tokens * bpt / 1048576
    return {
        "sampled": sample,
        "kv_estimate_bytes_per_token": bpt,
        "kv_baseline_mib": round(est_b, 2),
        "kv_optimized_mib": round(est_o, 2),
        "kv_delta_mib": round(est_b - est_o, 2),
        "label": "estimate",
        "active_model": settings["active_model"],
    }


def _baseline_window():
    history = state["raw_history"]
    budget = settings["max_context_tokens"]
    included, used, evicted = [], 0, []
    for h in reversed(history):
        if used + h["tokens"] <= budget:
            included.append(h)
            used += h["tokens"]
        else:
            evicted.append(h)
    included.reverse()
    evicted.reverse()
    return included, evicted, used


def _naive_probe(query: str):
    included, _, _ = _baseline_window()
    scored = []
    for h in included:
        sim = _token_overlap(query, h["user"])
        if sim > 0:
            scored.append({"turn_id": h["turn_id"], "excerpt": h["user"][:90], "sim": round(sim, 2)})
    scored.sort(key=lambda x: x["sim"], reverse=True)
    return scored[:5]


WINDOW_REPLAY_PROMPT = (
    "You are an assistant whose ONLY source of information is the conversation transcript below. "
    "The transcript may be incomplete because older turns were truncated to fit a context budget.\n\n"
    "From the transcript: 1) Answer the user's latest message using ONLY facts present in the transcript. "
    "If the transcript does not contain enough context to answer, reply exactly NO_LOCAL_ANSWER. "
    "2) List every factual claim visible in the transcript as short standalone facts.\n\n"
    "Be terse: answer in at most 12 words, each fact at most 10 words.\n\n"
    "Return strict JSON only: {\"answer\": \"...\", \"facts\": [\"fact1\", \"fact2\", ...]}"
)


def _llm_window_replay(included_turns: list) -> dict:
    if not included_turns:
        return {"answer": "", "facts_seen": [], "raw": "",
                "prompt_eval_count": 0, "eval_count": 0, "ms": 0}
    transcript = "\n".join(f'user (turn {h["turn_id"]}): {h["user"]}' for h in included_turns)
    prompt = (f"{WINDOW_REPLAY_PROMPT}\n\n--- BEGIN TRANSCRIPT ---\n{transcript}\n"
              f"--- END TRANSCRIPT ---\n\nReturn JSON:")
    try:
        resp = requests.post(f"{ollama_base}/api/generate", json={
            "model": settings["active_model"],
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_predict": 300, "num_ctx": 8192},
        }, timeout=180)
        data = resp.json()
        raw = str(data.get("response", ""))
        try:
            out = json.loads(raw)
        except Exception:
            out = {}
        return {
            "answer": str(out.get("answer") or ""),
            "facts_seen": [str(f) for f in (out.get("facts") or [])][:40],
            "raw": raw,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "ms": (resp.elapsed.total_seconds() * 1000) if hasattr(resp, "elapsed") else None,
        }
    except Exception as e:
        return {"answer": "", "facts_seen": [], "raw": "", "error": str(e),
                "prompt_eval_count": None, "eval_count": None, "ms": None}


ANSWER_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the context memory entries below. "
    "If the context entries do not contain the answer, reply exactly NO_LOCAL_ANSWER. "
    "Answer in a short sentence of at most 15 words."
)


def _llm_answer(query: str, context_facts: list) -> dict:
    if not query:
        return {"answer": "", "prompt_eval_count": 0, "eval_count": 0, "ms": 0}
    context = "\n".join(f"- {f['fact']}" for f in (context_facts or [])) or "(no context entries)"
    prompt = f"{ANSWER_PROMPT}\n\n--- CONTEXT ---\n{context}\n---\n\nQuestion: {query}\nAnswer:"
    try:
        resp = requests.post(f"{ollama_base}/api/generate", json={
            "model": settings["active_model"],
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_predict": 80, "num_ctx": 8192},
        }, timeout=180)
        data = resp.json()
        return {
            "answer": str(data.get("response", "") or "").strip(),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "ms": (resp.elapsed.total_seconds() * 1000) if hasattr(resp, "elapsed") else None,
        }
    except Exception:
        return {"answer": "", "prompt_eval_count": None, "eval_count": None, "ms": None}


def _content_tokens(prompt_eval_count, constant) -> int:
    if prompt_eval_count is None or constant is None:
        return None
    return max(0, prompt_eval_count - constant)


def _build_comparison(context_facts: list = None, probe_query: str = None, full: bool = False) -> dict:
    if context_facts is None:
        context_facts = state["last_context"].get("facts", [])
    history = state["raw_history"]
    trace = state["active_token_trace"]
    inj_trace = state["inject_token_trace"]

    cumulative_raw = 0
    token_history = []
    for i, h in enumerate(history):
        cumulative_raw += h["tokens"]
        token_history.append({
            "turn_id": h["turn_id"],
            "raw_cumulative": cumulative_raw,
            "active_tokens": trace[i] if i < len(trace) else 0,
            "injected_tokens": inj_trace[i] if i < len(inj_trace) else 0,
        })

    included, evicted, used = _baseline_window()

    last_replay = state["baseline_replay"][-1] if state["baseline_replay"] else None
    measured_win = last_replay.get("window_tokens") if last_replay else None
    baseline_tokens = measured_win if measured_win is not None else used

    active_measured = [m.get("measured_tokens") for m in state["active_memories"]]
    unmeasured_active = sum(1 for t in active_measured if t is None)
    active_tokens = sum(t for t in active_measured if t is not None)

    context_tokens = state["last_context"].get("injected_tokens") if state.get("last_context") else None

    evicted_ids = [h["turn_id"] for h in evicted]
    evicted_turn_ids_set = set(evicted_ids)

    gt_by_turn = {}
    for gt in state["ground_truth"]:
        gt_by_turn.setdefault(gt["source_turn"], []).append(gt)
    evicted_gt_turns = set(gt_by_turn) & evicted_turn_ids_set

    def _held(ev_turn, fact_text):
        for m in state["active_memories"]:
            if m.get("source_turn_id") == ev_turn:
                return m
            if SequenceMatcher(None, fact_text.lower(), m["fact"].lower()).ratio() >= 0.7:
                return m
        return None

    forgot_but_kept = []
    forgot_by_both = 0
    for ev_turn in sorted(evicted_gt_turns):
        for gt in gt_by_turn.get(ev_turn, []):
            mem = _held(ev_turn, gt["fact"])
            if mem is not None:
                forgot_but_kept.append({
                    "fact": gt["fact"],
                    "category": gt["category"],
                    "source_turn": ev_turn,
                    "current_importance": round(mem.get("current_importance", mem.get("base_score", 0)), 3),
                    "is_planted": True,
                })
            else:
                forgot_by_both += 1

    kept_extra_other = []
    for mem in state["active_memories"]:
        if mem.get("source_turn_id") in evicted_turn_ids_set - evicted_gt_turns:
            kept_extra_other.append({
                "fact": mem["fact"],
                "category": mem["category"],
                "source_turn": mem["source_turn_id"],
                "current_importance": round(mem.get("current_importance", mem.get("base_score", 0)), 3),
            })

    baseline_memory_preview = {
        "included_turns": [{"turn_id": h["turn_id"], "excerpt": h["user"][:90], "tokens": h["tokens"]} for h in included],
        "evicted_turn_ids": evicted_ids,
        "evicted_tokens": sum(h["tokens"] for h in evicted),
        "planted_facts_in_window": sum(
            1 for gt in state["ground_truth"] if gt["source_turn"] not in evicted_turn_ids_set
        ),
        "planted_facts_in_window_vs_evicted": {
            "in_window": sum(1 for gt in state["ground_truth"] if gt["source_turn"] not in evicted_turn_ids_set),
            "evicted": sum(1 for gt in state["ground_truth"] if gt["source_turn"] in evicted_turn_ids_set),
        },
        "planted_facts_total": len(state["ground_truth"]),
    }

    def collect_window_facts(turns, cap):
        out, seen = [], set()
        for h in turns:
            for f in h.get("facts", []):
                text = f.get("fact", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append({
                    "fact": text,
                    "category": f.get("category", ""),
                    "source_turn": f.get("source_turn_id", h["turn_id"]),
                })
                if len(out) >= cap:
                    return out
        return out

    baseline_memory_summary = {
        "remembered": collect_window_facts(included, 5000 if full else 200),
        "lost": collect_window_facts(evicted, 20000 if full else 500),
        "remembered_turns": len(included),
        "lost_turns": len(evicted),
        "remembered_unbounded": sum(len(h.get("facts", [])) for h in included),
        "lost_unbounded": sum(len(h.get("facts", [])) for h in evicted),
    }

    return {
        "system_token_budget": settings["max_context_tokens"],
        "cumulative_history_tokens": cumulative_raw,
        "baseline_context_tokens": baseline_tokens,
        "baseline_context_tokens_measured": measured_win is not None,
        "baseline_turns_in_context": len(included),
        "baseline_evicted_turn_count": len(evicted),
        "baseline_evicted_turn_ids": evicted_ids if full else evicted_ids[:200],
        "baseline_evicted_tokens": sum(h["tokens"] for h in evicted),
        "active_memory_tokens": active_tokens,
        "active_memory_count": len(state["active_memories"]),
        "active_memory_unmeasured_count": unmeasured_active,
        "pruned_memory_count": len(state["pruned_memories"]),
        "adaptive_budget_cap": settings["max_context_tokens"],
        "budget_squeezed": bool(state["last_budget_evictions"]),
        "adaptive_budget_evictions": state["last_budget_evictions"],
        "injection_token_limit": settings["injection_token_limit"],
        "retrieval_engines": state["last_context"].get("engines", []),
        "optimized_context_tokens": context_tokens,
        "optimized_facts_in_context": len(context_facts or []),
        "metric_source": "live model prompt_eval_count (content tokens, measured live)",
        "facts_barebones_forgot_you_still_keep": forgot_but_kept,
        "forgot_by_both": forgot_by_both,
        "kept_extra_other": kept_extra_other,
        "baseline_memory_preview": baseline_memory_preview,
        "baseline_memory_summary": baseline_memory_summary,
        "baseline_naive_probe": _naive_probe(probe_query) if probe_query else None,
        "token_history": token_history,
        "latency_stats": state["latency_stats"],
        "settings": settings,
        "vram": _vram_block(used, active_tokens),
    }


def _record_latency(stage: str, ms: float):
    state["latency_stats"].setdefault(stage, []).append(ms)


def _record_timeline(t_id, user_message, facts, context_facts, pruned_this_turn, injected_tokens,
                     baseline_answer=None, optimized_answer=None, facts_seen=None):
    included, evicted, used = _baseline_window()
    evicted_ids = [h["turn_id"] for h in evicted]
    last = set(state["last_evicted_ids"])
    newly = [i for i in evicted_ids if i not in last][-50:]
    state["last_evicted_ids"] = evicted_ids[-200:]

    newly_lost = []
    for mem in state["active_memories"]:
        if mem.get("source_turn_id") in set(newly):
            newly_lost.append({
                "fact": mem["fact"],
                "category": mem["category"],
                "current_importance": round(mem.get("current_importance", mem.get("base_score", 0)), 3),
            })

    state["timeline"].append({
        "turn_id": t_id,
        "user_message": user_message,
        "extracted_facts": [dict(f) for f in facts],
        "optimized_retrieved_this_turn": [dict(f) for f in (context_facts or [])],
        "optimized_pruned_this_turn": [dict(f) for f in pruned_this_turn],
        "optimized_injected_tokens": injected_tokens,
        "optimized_budget_squeezed": bool(state["last_budget_evictions"]),
        "optimized_budget_evictions": state["last_budget_evictions"],
        "baseline_window_turn_ids": [h["turn_id"] for h in included][-30:],
        "baseline_window_tokens": used,
        "baseline_newly_evicted_turn_ids": newly,
        "baseline_newly_lost_facts": newly_lost,
        "baseline_answer": baseline_answer or "",
        "optimized_answer": optimized_answer or "",
        "baseline_facts_seen": facts_seen or [],
    })
    state["timeline"] = state["timeline"][-10000:]


def _ingest_turn(user_message: str, facts: list, external_turn_id: int = None, replay: bool = False,
                 real_msg_tokens: int = None) -> dict:
    if external_turn_id is None:
        state["current_turn"] += 1
        t_id = state["current_turn"]
    else:
        state["current_turn"] = external_turn_id
        t_id = external_turn_id

    t0 = time.perf_counter()
    new_facts = []
    for item in facts:
        item["source_turn_id"] = item.get("source_turn_id", t_id)
        item["last_access_turn"] = t_id
        item["base_score"] = scorer.compute_score(item, query_relevance=0.8, current_turn=t_id)
        new_facts.append(item.copy())
    _record_latency("scoring", (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    for item in new_facts:
        measured = _fact_tokens(item["fact"])
        if measured is not None:
            item["measured_tokens"] = measured
    _record_latency("fact_token_measurement", (time.perf_counter() - t0) * 1000)

    if settings["enable_compression"]:
        t0 = time.perf_counter()
        state["active_memories"] = compressor.dedupe_incremental(state["active_memories"], new_facts,
                                                                 embed_fn=_embed_texts)
        _record_latency("compression", (time.perf_counter() - t0) * 1000)
    else:
        state["active_memories"].extend(new_facts)

    t0 = time.perf_counter()
    active, pruned_this_turn = decay_engine.step_decay_and_prune(state["active_memories"], t_id)
    _record_latency("decay", (time.perf_counter() - t0) * 1000)
    state["active_memories"] = active
    state["pruned_memories"].extend(pruned_this_turn)

    t0 = time.perf_counter()
    kept, evicted_by_budget, _ = token_budget_evict(
        state["active_memories"], settings["max_context_tokens"], _lookup_or_measure)
    state["active_memories"] = kept
    state["last_budget_evictions"] = [{
        "fact": m["fact"],
        "category": m.get("category", ""),
        "current_importance": round(m.get("current_importance", 0), 3),
        "source_turn_id": m.get("source_turn_id"),
    } for m in evicted_by_budget]
    _record_latency("budget_evict", (time.perf_counter() - t0) * 1000)

    if real_msg_tokens is None:
        real_msg_tokens = _fact_tokens(user_message)
    msg_tokens = real_msg_tokens if real_msg_tokens is not None else 1

    state["raw_history"].append({
        "turn_id": t_id,
        "user": user_message,
        "tokens": msg_tokens,
        "token_measured": real_msg_tokens is not None,
        "facts": [dict(f) for f in facts],
    })

    active_tokens = sum(m.get("measured_tokens") for m in state["active_memories"] if m.get("measured_tokens") is not None)
    state["active_token_trace"].append(active_tokens)

    t0 = time.perf_counter()
    context_facts = retriever.retrieve(user_message, state["active_memories"], current_turn=t_id,
                                       token_limit=settings["injection_token_limit"],
                                       fact_tokens=_lookup_or_measure)
    if retriever.stats.get("embed_ms"):
        _record_latency("embedding_ms", retriever.stats["embed_ms"])
    _record_latency("retrieval", (time.perf_counter() - t0) * 1000)

    inject_trace_val = state["last_context"].get("injected_tokens") if state.get("last_context") else None
    state["inject_token_trace"].append(inject_trace_val)

    baseline_answer, optimized_answer, facts_seen = "", "", []
    window_tokens, window_turn_ids = None, []
    injected_tokens = None
    if replay:
        included, _, _ = _baseline_window()
        window_turn_ids = [h["turn_id"] for h in included]
        constants = _calibrated_constants()
        t0 = time.perf_counter()
        bl = _llm_window_replay(included)
        _record_latency("window_replay", (time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        ans = _llm_answer(user_message, context_facts)
        _record_latency("answer", (time.perf_counter() - t0) * 1000)
        window_tokens = _content_tokens(bl.get("prompt_eval_count"), constants.get("C_replay"))
        injected_tokens = _content_tokens(ans.get("prompt_eval_count"), constants.get("C_answer"))
        if injected_tokens is None and context_facts:
            context_text = "\n".join(f["fact"] for f in context_facts)
            injected_tokens = _fact_tokens(context_text)
        if window_tokens is None and included:
            transcript = "\n".join(f'user (turn {h["turn_id"]}): {h["user"]}' for h in included)
            window_tokens = _fact_tokens(transcript)
        baseline_answer = bl.get("answer", "")
        facts_seen = bl.get("facts_seen", [])
        optimized_answer = ans.get("answer", "")
        state["baseline_replay"].append({
            "turn_id": t_id,
            "query": user_message,
            "baseline_answer": baseline_answer,
            "optimized_answer": optimized_answer,
            "facts_seen": facts_seen,
            "window_tokens": window_tokens,
            "window_turn_ids": window_turn_ids,
            "injected_tokens": injected_tokens,
            "window_eval_count": bl.get("eval_count"),
            "answer_eval_count": ans.get("eval_count"),
            "window_latency_ms": bl.get("ms"),
            "answer_latency_ms": ans.get("ms"),
            "window_turn_count": len(window_turn_ids),
        })

    state["last_context"] = {
        "query": user_message,
        "facts": [dict(f) for f in context_facts],
        "injected_tokens": injected_tokens,
        "token_measured": injected_tokens is not None,
        "engines": sorted({f.get("retrieval_engine", "") for f in context_facts if f.get("retrieval_engine")}),
    }

    _record_timeline(t_id, user_message, facts, context_facts, pruned_this_turn, injected_tokens,
                     baseline_answer=baseline_answer, optimized_answer=optimized_answer, facts_seen=facts_seen)
    return _build_comparison(context_facts, probe_query=None)


FILLERS = {
    "short": (" Routine monitoring shows nothing unusual; the log pipelines are "
              "stable and the refactor is progressing on schedule without conflicts."),
    "normal": (" The overnight batches all finished clean; nothing unusual surfaced in the log "
               "pipelines, the deployment target is stable, and the refactor work we scheduled "
               "is progressing exactly as planned without any unhandled conflicts or open questions "
               "that would need escalation."),
    "long": (" The overnight batches all finished clean and nothing unusual surfaced in the log "
             "pipelines; the deployment target for the current sprint is stable with no pending "
             "rollback needed, and the refactor work we scheduled earlier is progressing exactly "
             "as planned without any unhandled conflicts, flaky tests, or open questions that would "
             "need escalation, so the team can stay on the committed timeline risk-free."),
}


SEED_MESSAGES = [
    (2, "personal",
     "Heads up, remember this: I have a cat named Whiskers. Please remember my cat's name and that she loves tuna.",
     "User has a cat named Whiskers"),
    (0.10, "project_context",
     "Please note for the project: key_alpha_legacy is a config key my core project depends on, so remember it stays on.",
     "User core project: config key_alpha_legacy is on"),
    (0.50, "project_context",
     "Architecture decision under discussion: each feature's flag key will segment the rollout. Remember this project decision.",
     "Project architecture decision under discussion: flag keys segment the rollout"),
    (0.90, "project_context",
     "Project update: the feature flag config key_demo_90 is enabled for the rollout. Please remember that key_demo_90 is enabled.",
     "Project feature flag config key_demo_90 is enabled"),
    (0.95, "technical_preference",
     "Important requirement: the production cluster must run PostgreSQL 15 as the primary database. No exceptions, remember this.",
     "User primary DB requirement: must use PostgreSQL 15"),
]


def _generated_planted(n: int):
    """Deterministic extra planted facts beyond the 5 hand-written seeds."""
    cats = ["project_context", "technical_preference", "personal"]
    c = cats[(n - 5) % len(cats)]
    if c == "project_context":
        fact = f"Project feature flag config key_demo_r{n} is enabled for the rollout"
        msg = (f"Project note: the rollout flag config key_demo_r{n} is being enabled. "
               f"Remember that config key_demo_r{n} stays on.")
    elif c == "technical_preference":
        fact = f"User primary DB requirement: staging cluster runs PostgreSQL 15 instance db_{n}"
        msg = (f"Requirement: the staging cluster primary database must be PostgreSQL 15 "
               f"(instance db_{n}). No exceptions.")
    else:
        fact = f"User has a recurring scheduled job that must run daily: daily_{n}"
        msg = f"Remember the daily scheduled job daily_{n} must keep running."
    return c, msg, fact


def _build_demo_messages(turns: int = 150, density_pct: float = 20.0,
                         planted_count: int = 5, filler: str = "normal"):
    num_turns = max(20, min(500, int(turns)))
    filler_text = FILLERS.get(str(filler), FILLERS["normal"])
    density_pct = max(0.0, min(100.0, float(density_pct)))

    seed = int(cfg["system"].get("seed", 42))
    rng = random.Random(seed * 1009 ^ num_turns * 31 ^ int(planted_count) * 17 ^ int(density_pct))

    planted = max(1, min(int(planted_count), num_turns - 19))
    base_facts = [m[1:] for m in SEED_MESSAGES]

    if planted == 1:
        positions = [2]
    else:
        positions = {2}
        for i in range(1, planted):
            positions.add(min(num_turns, 3 + (num_turns - 3) * i // (planted - 1)))
        positions = sorted(positions)
    if positions and positions[-1] == num_turns and len(positions) > 1:
        positions[-1] = num_turns - 1

    planted_facts = []
    for idx, pos in enumerate(positions):
        if idx < len(base_facts):
            p_cat, p_msg, p_fact = base_facts[idx]
        else:
            p_cat, p_msg, p_fact = _generated_planted(idx + 1)
        planted_facts.append((pos, p_cat, p_msg, p_fact))

    available = [t for t in range(3, num_turns + 1) if t not in set(positions)]
    n_density = int(round(num_turns * density_pct / 100.0))
    rng.shuffle(available)
    density_turns = set(available[:n_density])
    density_map = {t: planted_facts[i % len(planted_facts)] for i, t in enumerate(sorted(density_turns))}

    messages, ground_truth = [], []
    planted_pos = {p[0] for p in planted_facts}
    for turn_id in range(1, num_turns + 1):
        if turn_id in planted_pos:
            _, p_cat, p_msg, p_fact = next(p for p in planted_facts if p[0] == turn_id)
            messages.append({"turn_id": turn_id, "user": p_msg + filler_text})
            ground_truth.append({"source_turn": turn_id, "fact": p_fact, "category": p_cat, "is_trap": False})
        elif turn_id in density_map:
            _, _, _, fact = density_map[turn_id]
            messages.append({
                "turn_id": turn_id,
                "user": (f"Turn {turn_id}: Discussing routine system log outputs and generic code refactoring."
                         f"{filler_text} Reconfirming an earlier note: {fact}."),
            })
        else:
            messages.append({
                "turn_id": turn_id,
                "user": f"Turn {turn_id}: Discussing routine system log outputs and generic code refactoring.{filler_text}",
            })
    return messages, ground_truth


def _reset_demo_state():
    state["current_turn"] = 0
    state["active_memories"] = []
    state["pruned_memories"] = []
    state["raw_history"] = []
    state["active_token_trace"] = []
    state["inject_token_trace"] = []
    state["latency_stats"] = {}
    state["timeline"] = []
    state["last_evicted_ids"] = []
    state["ground_truth"] = []
    state["baseline_replay"] = []
    state["last_context"] = {}


def _run_llm_demo(messages):
    job = state["demo_job"]
    job["running"] = True
    job["started_at"] = time.time()
    total = len(messages)
    try:
        for i, m in enumerate(messages, start=1):
            if job.get("cancel_requested"):
                job["cancelled"] = True
                break
            t0 = time.perf_counter()
            extracted, emeta = extractor.extract_facts(m["turn_id"], m["user"])
            _record_latency("extraction", (time.perf_counter() - t0) * 1000)
            real_msg_tokens = _extraction_content_tokens(emeta)
            if job.get("cancel_requested"):
                job["cancelled"] = True
                break
            job["extraction_fallback"] = job.get("extraction_fallback", 0) + (1 if emeta.get("fallback") else 0)
            _ingest_turn(m["user"], extracted, external_turn_id=m["turn_id"], replay=True,
                         real_msg_tokens=real_msg_tokens)
            job["turns_done"] = i
            job["current_turn"] = m["turn_id"]
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["running"] = False
        job["finished_at"] = time.time()
        job["turns_total"] = total


@app.post("/api/chat")
def chat_endpoint(msg: ChatMessage):
    t_id = state["current_turn"] + 1
    t0 = time.perf_counter()
    extracted, emeta = extractor.extract_facts(t_id, msg.user_message)
    _record_latency("extraction", (time.perf_counter() - t0) * 1000)
    real_msg_tokens = _extraction_content_tokens(emeta)
    comparison = _ingest_turn(msg.user_message, extracted, replay=True, real_msg_tokens=real_msg_tokens)
    retrieved_context = retriever.retrieve(msg.user_message, state["active_memories"])
    last = state["baseline_replay"][-1] if state["baseline_replay"] else {}

    return {
        "turn_id": state["current_turn"],
        "bot_response": f"Processed turn {state['current_turn']}. Contextually retrieved {len(retrieved_context)} relevant memories.",
        "optimized_answer": last.get("optimized_answer", ""),
        "baseline_answer": last.get("baseline_answer", ""),
        "baseline_facts_seen": last.get("facts_seen", []),
        "active_memories": state["active_memories"],
        "pruned_memories": state["pruned_memories"],
        "retrieved_context": retrieved_context,
        "comparison": comparison,
    }


@app.post("/api/demo/scenario")
def demo_scenario(payload: dict = None):
    payload = payload or {}
    turns = max(20, min(500, int(payload.get("turns", 150))))
    density_pct = max(0.0, min(100.0, float(payload.get("density_pct", 20.0))))
    planted_count = max(1, min(50, int(payload.get("planted_count", 5))))
    filler = str(payload.get("filler", "normal"))

    if state["demo_job"].get("running"):
        return {"status": "busy", "detail": "A demo is already running", "job": state["demo_job"]}

    _reset_demo_state()
    state["last_demo"] = {
        "turns": turns, "density_pct": density_pct, "planted_count": planted_count,
        "filler": filler, "mode": "llm",
    }
    state["demo_job"] = {
        "running": False, "mode": "llm", "turns_total": turns, "turns_done": 0,
        "current_turn": None, "cancel_requested": False, "cancelled": False,
        "error": None, "started_at": None, "finished_at": None, "eta_s": None,
    }

    messages, ground_truth = _build_demo_messages(turns, density_pct, planted_count, filler)
    state["ground_truth"] = ground_truth
    state["demo_job"].update(running=True, turns_total=len(messages), started_at=time.time())
    threading.Thread(target=_run_llm_demo, args=(messages,), daemon=True).start()
    return {"status": "started", "job": state["demo_job"]}


@app.get("/api/demo/job")
def demo_job():
    j = state["demo_job"]
    if j.get("running") and j.get("started_at"):
        elapsed = time.time() - j["started_at"]
        done = j.get("turns_done", 0) or 0
        per = elapsed / max(1, done)
        j["eta_s"] = int(per * max(0, j.get("turns_total", 0) - done))
        j["elapsed_s"] = int(elapsed)
    return {"job": j}


@app.post("/api/demo/job/cancel")
def cancel_demo_job():
    if state["demo_job"].get("running"):
        state["demo_job"]["cancel_requested"] = True
        return {"status": "cancel_requested", "job": state["demo_job"]}
    return {"status": "not_running", "job": state["demo_job"]}


@app.get("/api/timeline")
def get_timeline():
    tl = state["timeline"]
    return {"timeline": tl[-300:], "total": len(tl)}


@app.get("/api/timeline/full")
def get_timeline_full(offset: int = 0, limit: int = 50):
    tl = state["timeline"]
    offset = max(0, int(offset))
    limit = max(1, min(200, int(limit)))
    return {"total": len(tl), "offset": offset, "limit": limit, "entries": tl[offset:offset + limit]}


@app.get("/api/comparison/full")
def comparison_full():
    return _build_comparison(full=True)


@app.get("/api/experiments/latest")
def experiments_latest():
    p = BASE_DIR / "experiments/results/live_manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"detail": "No experiments have been run yet.", "found": False}


@app.get("/api/state")
def get_state():
    return {
        "current_turn": state["current_turn"],
        "active_memories": state["active_memories"],
        "pruned_memories": state["pruned_memories"],
        "ground_truth_count": len(state["ground_truth"]),
        "ground_truth": [{
            "source_turn": g["source_turn"],
            "fact": g["fact"],
            "category": g["category"],
            "is_trap": g.get("is_trap", False),
        } for g in state["ground_truth"]],
        "comparison": _build_comparison(),
        "settings": settings,
        "demo_params": state.get("last_demo", {}),
        "demo_job": state["demo_job"],
        "baseline_replay": state["baseline_replay"][-200:],
        "baseline_replay_count": len(state["baseline_replay"]),
    }


@app.get("/api/models")
def get_models():
    try:
        tags = requests.get(f"{ollama_base}/api/tags", timeout=5).json().get("models", [])
    except Exception:
        tags = []
    models = []
    for m in tags:
        name = m["name"]
        if "embed" in name.lower():
            continue
        details = m.get("details", {})
        models.append({
            "name": name,
            "params": details.get("parameter_size", ""),
            "family": (details.get("families") or [""])[0],
        })
    return {"models": models, "active": settings["active_model"]}


@app.post("/api/models/select")
def select_model(payload: dict):
    name = payload.get("model")
    if not name:
        return {"status": "error", "detail": "no model"}
    prev = settings["active_model"]
    if prev != name:
        try:
            requests.post(f"{ollama_base}/api/generate",
                          json={"model": prev, "prompt": "x", "keep_alive": 0},
                          timeout=10)
        except Exception:
            pass
    settings["active_model"] = name
    extractor.model = name
    _token_constants.update(model=None)
    _fact_token_cache.clear()
    _persist_settings()
    return {"status": "ok", "active": name, "unloaded_previous": prev if prev != name else None}


@app.get("/api/settings")
def get_settings():
    return settings


class SettingsPayload(BaseModel):
    max_context_tokens: int = None
    top_k: int = None
    similarity_threshold: float = None
    pruning_threshold: float = None
    enable_compression: bool = None
    injection_token_limit: int = None


@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    if payload.max_context_tokens is not None:
        settings["max_context_tokens"] = max(256, int(payload.max_context_tokens))
    if payload.top_k is not None:
        settings["top_k"] = max(1, int(payload.top_k))
        retriever.top_k = settings["top_k"]
    if payload.similarity_threshold is not None:
        settings["similarity_threshold"] = float(payload.similarity_threshold)
        retriever.sim_threshold = settings["similarity_threshold"]
    if payload.pruning_threshold is not None:
        settings["pruning_threshold"] = float(payload.pruning_threshold)
        decay_engine.pruning_threshold = settings["pruning_threshold"]
    if payload.enable_compression is not None:
        settings["enable_compression"] = bool(payload.enable_compression)
    if payload.injection_token_limit is not None:
        settings["injection_token_limit"] = max(0, int(payload.injection_token_limit))
    _persist_settings()
    return {"status": "ok", "settings": settings, "comparison": _build_comparison()}


@app.get("/api/gpu")
def gpu_endpoint():
    return {"vram": _vram_block(
        _build_comparison()["baseline_context_tokens"],
        _build_comparison()["active_memory_tokens"],
    )}


@app.post("/api/experiments/run")
def run_experiments():
    from experiments.live import compute_all_json

    if state["demo_job"].get("running"):
        return {"status": "error", "detail": "Demo is still running; wait for it to finish first"}
    if not state["raw_history"]:
        return {"status": "error", "detail": "No demo loaded yet. Run a demo first."}

    stream = [{
        "turn_id": h["turn_id"],
        "user": h["user"],
        "tokens": h.get("tokens"),
        "facts": [dict(f) for f in h.get("facts", [])],
    } for h in state["raw_history"]]
    ground_truth = state["ground_truth"]

    results = compute_all_json(state, settings, stream, ground_truth, scorer, decay_engine,
                               retriever, _build_comparison, fact_tokens=_lookup_or_measure,
                               embed_fn=_embed_texts, embedding_model=embed_model,
                               injection_token_limit=settings["injection_token_limit"])

    out_path = BASE_DIR / "experiments/results/live_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(out_path, "w") as f:
        _json.dump(results, f, indent=2, default=str)
    return results


@app.get("/api/ethics/privacy_export")
def privacy_export():
    return {
        "user_data_retention_policy": "Zero-Knowledge Local Storage",
        "active_memories": state["active_memories"],
        "pruned_memories": state["pruned_memories"],
    }


@app.delete("/api/ethics/purge")
def privacy_purge():
    state["active_memories"].clear()
    state["pruned_memories"].clear()
    state["raw_history"].clear()
    state["active_token_trace"].clear()
    state["inject_token_trace"].clear()
    state["ground_truth"].clear()
    state["latency_stats"] = {}
    state["timeline"].clear()
    state["last_evicted_ids"] = []
    state["current_turn"] = 0
    state["last_context"] = {}
    state["baseline_replay"] = []
    state["last_budget_evictions"] = []
    return {"status": "All stored user memory structures successfully purged."}


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    with open(BASE_DIR / "ui/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)