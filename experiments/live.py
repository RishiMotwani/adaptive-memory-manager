"""Live-computed experiments (E1-E6) evaluated against the same in-memory test case for
both the proposed system and the barebones baseline. Pure functions over state; the only
external calls are the shared token/embedding measurement functions passed in (fact_tokens,
embed_fn), so E5 replays the same semantics as the live pipeline."""

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.power import TTestIndPower

from memory_optimizer.scoring import ImportanceScorer
from memory_optimizer.decay import CategoryDecayEngine
from memory_optimizer.retrieval import MemoryRetriever
from memory_optimizer.compression import MemoryCompressor
from memory_optimizer.budget import token_budget_evict


def _overlap(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    exact = a_set.intersection(b_set)
    if not exact:
        exact = {x for x in a_set for y in b_set if x in y or y in x}
    return len(exact) / (max(len(a_set), 1))


def _recall_proposed(ground_truth: list, active_memories: list) -> float:
    if not ground_truth:
        return 0.0
    hits = 0
    for gt in ground_truth:
        for m in active_memories:
            if m.get("source_turn_id") == gt["source_turn"] or _overlap(gt["fact"], m["fact"]) >= 0.7:
                hits += 1
                break
    return hits / len(ground_truth)


def _recall_baseline(ground_truth: list, included_turn_ids: set) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for gt in ground_truth if gt["source_turn"] in included_turn_ids)
    return hits / len(ground_truth)


def _recall_baseline_real(ground_truth: list, replay: list) -> float:
    if not ground_truth or not replay:
        return None
    facts_seen = set(replay[-1].get("facts_seen") or [])
    if not facts_seen:
        return None
    hits = sum(1 for gt in ground_truth if any(_overlap(gt["fact"], f) >= 0.6 for f in facts_seen))
    return hits / len(ground_truth)


def _forgetting_precision(ground_truth: list, forgotten_turn_ids: set, categories=("transient",)) -> float:
    relevant = [gt for gt in ground_truth if gt["category"] in categories]
    if not relevant:
        return 0.0
    correct = sum(1 for gt in relevant if gt["source_turn"] in forgotten_turn_ids)
    return correct / len(relevant)


def _is_int(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def _e1(token_history: list) -> dict:
    pairs = [(p.get("turn_id", i), p["injected_tokens"], p.get("baseline_context"))
             for i, p in enumerate(token_history)
             if _is_int(p["injected_tokens"]) and _is_int(p.get("baseline_context"))]
    proposed = [a for _, a, _ in pairs]
    baseline = [b for _, _, b in pairs]
    result = {
        "proposed_mean_tokens": float(np.mean(proposed)) if proposed else 0,
        "baseline_mean_tokens": float(np.mean(baseline)) if baseline else 0,
        "n": len(proposed),
        "metric_source": "live model prompt_eval_count (content tokens, measured live)",
        "baseline_measured_live": True,
    }
    if len(set(proposed)) <= 1 or len(set(baseline)) <= 1:
        result.update({"p_value": float("nan"), "significant": None})
    else:
        stat, p_val = wilcoxon(proposed, baseline)
        result.update({"p_value": float(p_val), "significant": bool(p_val < 0.05)})
    result["per_turn"] = [
        {"turn": tid, "proposed": a, "baseline": b} for tid, a, b in pairs
    ]
    return result


def _e2(ground_truth: list, state: dict, included_turn_ids: set, pruned_turn_ids: set) -> dict:
    real = _recall_baseline_real(ground_truth, state.get("baseline_replay", []))
    return {
        "proposed_positive_recall": round(_recall_proposed(ground_truth, state["active_memories"]), 3),
        "baseline_positive_recall": round(real if real is not None
                                          else _recall_baseline(ground_truth, included_turn_ids), 3),
        "baseline_recall_real": real is not None,
        "proposed_forgetting_precision": round(
            _forgetting_precision(ground_truth, pruned_turn_ids), 3),
        "baseline_forgetting_precision": round(
            _forgetting_precision(ground_truth, set(included_turn_ids) - set(range(0)), categories=("transient",)), 3)
        if ground_truth else 0.0,
    }


def _e3(latency_stats: dict) -> dict:
    def avg(ms_list):
        return round(float(np.mean(ms_list)), 2) if ms_list else None
    return {
        "extraction_ms": avg(latency_stats.get("extraction", [])),
        "scoring_ms": avg(latency_stats.get("scoring", [])),
        "fact_token_measurement_ms": avg(latency_stats.get("fact_token_measurement", [])),
        "compression_ms": avg(latency_stats.get("compression", [])),
        "decay_ms": avg(latency_stats.get("decay", [])),
        "budget_evict_ms": avg(latency_stats.get("budget_evict", [])),
        "retrieval_ms": avg(latency_stats.get("retrieval", [])),
        "embedding_ms": avg(latency_stats.get("embedding_ms", [])),
        "window_replay_ms": avg(latency_stats.get("window_replay", [])),
        "answer_ms": avg(latency_stats.get("answer", [])),
    }


def _e4(ground_truth: list, current_turn: int, active_memories: list, included_turn_ids: set, replay: list = None) -> dict:
    facts_seen = set((replay[-1].get("facts_seen") or []) if replay else [])
    all_turn_dists = [current_turn - gt["source_turn"] for gt in ground_truth if gt["source_turn"] < current_turn]
    if all_turn_dists:
        span = max(all_turn_dists)
    else:
        span = 0
    candidates = [5000, 2500, 1000, 500, 200, 100, 50, 10]
    distances = [d for d in candidates if span >= d and d <= current_turn - 5]
    if len(distances) < 2:
        distances = [d for d in [500, 200, 100, 50, 10] if d <= current_turn - 2]
    proposed, baseline = [], []
    for d in distances:
        relevant = [gt for gt in ground_truth if current_turn - gt["source_turn"] >= d]
        if not relevant:
            proposed.append(None)
            baseline.append(None)
            continue
        p_hits = sum(
            1 for gt in relevant
            if any(m.get("source_turn_id") == gt["source_turn"] or _overlap(gt["fact"], m["fact"]) >= 0.7
                   for m in active_memories)
        )
        if facts_seen:
            b_hits = sum(1 for gt in relevant if any(_overlap(gt["fact"], f) >= 0.6 for f in facts_seen))
        else:
            b_hits = sum(1 for gt in relevant if gt["source_turn"] in included_turn_ids)
        proposed.append(round(p_hits / len(relevant), 3))
        baseline.append(round(b_hits / len(relevant), 3))
    return {"distances": distances, "proposed_accuracy": proposed, "baseline_accuracy": baseline,
            "baseline_real": bool(facts_seen)}


def _sampled_stream(stream: list, cap: int) -> list:
    if len(stream) <= cap:
        return stream
    needles = sorted({e["turn_id"] for e in stream
                      for f in e.get("facts", []) if f.get("category") != "transient"})
    keep = set(needles)
    step = max(1, (len(stream) - len(keep)) // max(1, cap - len(keep)))
    idx = 0
    while len(keep) < cap and idx < len(stream):
        keep.add(stream[idx]["turn_id"])
        idx += step
    by_id = {e["turn_id"]: e for e in stream}
    return [by_id[i] for i in sorted(keep)]


def _e5_replay(stream: list, ground_truth: list, toggles: dict, fact_tokens=None,
               embed_fn=None, embedding_model="nomic-embed-text", injection_token_limit=0) -> dict:
    lambdas = toggles.get("lambdas")
    weights = toggles.get("weights")
    threshold = toggles.get("threshold", 0.20)
    dedupe = toggles.get("dedupe", True)
    budget = int(toggles.get("budget", 4096))

    scorer = ImportanceScorer(weights=weights if weights else None)
    decay = CategoryDecayEngine(lambdas=lambdas, pruning_threshold=threshold)
    retriever = MemoryRetriever(top_k=5, sim_threshold=0.35, embed_fn=embed_fn,
                                embedding_model=embedding_model)
    compressor = MemoryCompressor()

    memories = []
    pruned = []
    inject_tokens = []

    max_context = budget
    raw_tokens_total = 0

    for entry in stream:
        t_id = entry["turn_id"]
        raw = entry.get("tokens")
        if raw is None and fact_tokens is not None:
            raw = fact_tokens(entry["user"])
        raw_tokens_total += raw or 0
        new_items = []
        for item in entry["facts"]:
            it = dict(item)
            it["source_turn_id"] = t_id
            it["last_access_turn"] = t_id
            it["base_score"] = scorer.compute_score(it, query_relevance=0.8, current_turn=t_id)
            new_items.append(it)
        if dedupe:
            compressor.dedupe_incremental(memories, new_items, embed_fn=embed_fn)
        else:
            memories.extend(new_items)
        active, pr = decay.step_decay_and_prune(memories, t_id)
        memories, pruned = active, pruned + pr
        if fact_tokens is not None:
            memories, _, _ = token_budget_evict(memories, max_context, fact_tokens)
        injected = retriever.retrieve(entry["user"], memories, current_turn=t_id,
                                      token_limit=injection_token_limit if fact_tokens is not None else 0,
                                      fact_tokens=fact_tokens)
        if fact_tokens is not None:
            sizes = [fact_tokens(f["fact"]) for f in injected]
            inject_tokens.append(sum(s for s in sizes if s is not None))
        else:
            inject_tokens.append(len(injected))

    recalled = _recall_proposed(ground_truth, memories)
    mean_inject = float(np.mean(inject_tokens)) if inject_tokens else 0

    return {
        "accuracy": round(recalled, 3),
        "mean_injected_tokens": round(mean_inject, 2),
        "active_count": len(memories),
        "pruned_count": len(pruned),
        "raw_tokens_total_at_budget": min(raw_tokens_total, max_context),
        "all_transient_pruned": round(_forgetting_precision(ground_truth, {m.get("source_turn_id") for m in pruned}), 3),
    }


def _e5(stream: list, ground_truth: list, base: str = "full", fact_tokens=None,
        embed_fn=None, embedding_model="nomic-embed-text", injection_token_limit=0) -> dict:
    full = _e5_replay(stream, ground_truth, {}, fact_tokens, embed_fn, embedding_model, injection_token_limit)
    no_decay = _e5_replay(stream, ground_truth, {"lambdas": {"transient": 0.0, "personal": 0.0,
                                                              "technical_preference": 0.0, "project_context": 0.0}},
                          fact_tokens, embed_fn, embedding_model, injection_token_limit)
    no_compression = _e5_replay(stream, ground_truth, {"dedupe": False},
                                fact_tokens, embed_fn, embedding_model, injection_token_limit)
    equal_weights = _e5_replay(stream, ground_truth, {"weights": {
        "w1_relevance": 0.25, "w2_utility": 0.25, "w3_recency": 0.25, "w4_frequency": 0.25}},
        fact_tokens, embed_fn, embedding_model, injection_token_limit)

    base_mean = full["mean_injected_tokens"] or 1

    def row(label, result):
        return {
            "ablation": label,
            "accuracy": result["accuracy"],
            "token_overhead_vs_full": round(result["mean_injected_tokens"] / base_mean, 2),
            "pruned_count": result["pruned_count"],
        }

    return {
        "full_system": row("full_system", full),
        "no_decay": row("no_decay", no_decay),
        "no_compression": row("no_compression", no_compression),
        "equal_weight_scoring": row("equal_weight_scoring", equal_weights),
        "token_source": "measured (live model prompt_eval_count / per-fact measured tokens)",
    }


def _e6(proposed: list, baseline: list) -> dict:
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if len(proposed) < 2 or np.all(proposed == baseline):
        return {"observed_cohens_d": None, "required_n_per_group": None, "alpha": 0.05, "target_power": 0.80,
                "n_actual": len(proposed)}
    pooled_std = np.sqrt((np.var(proposed) + np.var(baseline)) / 2)
    d = float(np.mean(proposed) - np.mean(baseline)) / pooled_std if pooled_std else float("nan")
    req_n = None
    try:
        analysis = TTestIndPower()
        req_n = int(np.ceil(analysis.solve_power(effect_size=abs(d), alpha=0.05, power=0.80)))
    except Exception:
        req_n = None
    return {
        "observed_cohens_d": round(abs(d), 3),
        "required_n_per_group": req_n,
        "alpha": 0.05,
        "target_power": 0.80,
        "n_actual": len(proposed),
    }


def compute_all(state: dict, settings: dict, stream: list, ground_truth: list,
                scorer, decay_engine, retriever, comparison_builder, fact_tokens=None,
                embed_fn=None, embedding_model="nomic-embed-text", injection_token_limit=0) -> dict:
    included_turn_ids = {h["turn_id"] for h in _baseline_window_of(state, settings)[0]}
    pruned_turn_ids = {m.get("source_turn_id") for m in state["pruned_memories"]}

    token_history = list(comparison_builder()["token_history"]) if comparison_builder else []
    budget = settings["max_context_tokens"]

    replay_map = {r["turn_id"]: r.get("window_tokens") for r in state.get("baseline_replay", [])}
    sliding_fallback = _per_turn_baseline_tokens(state["raw_history"], budget)
    for i, p in enumerate(token_history):
        b = replay_map.get(p["turn_id"])
        if b is None and i < len(sliding_fallback):
            b = sliding_fallback[i]
        p["baseline_context"] = b if _is_int(b) else 0
        p["baseline_measured"] = p["turn_id"] in replay_map

    pairs = [(p["injected_tokens"], p["baseline_context"])
             for p in token_history if _is_int(p["injected_tokens"]) and _is_int(p["baseline_context"])]
    proposed_series = [a for a, _ in pairs]
    baseline_series = [b for _, b in pairs]

    replay = state.get("baseline_replay", [])
    e1 = _e1(token_history)
    e2 = _e2(ground_truth, state, included_turn_ids, pruned_turn_ids)
    e3 = _e3(state["latency_stats"])
    e4 = _e4(ground_truth, state["current_turn"], state["active_memories"], included_turn_ids, replay)
    e5 = _e5(_sampled_stream(stream, 800), ground_truth, fact_tokens=fact_tokens,
             embed_fn=embed_fn, embedding_model=embedding_model, injection_token_limit=injection_token_limit)
    e6 = _e6(proposed_series, baseline_series)

    return {
        "model_under_test": settings["active_model"],
        "max_context_tokens": settings["max_context_tokens"],
        "metric_source": "live model prompt_eval_count (content tokens), same measurement method on both sides",
        "E1_token_efficiency": e1,
        "E2_memory_accuracy": e2,
        "E3_latency": e3,
        "E4_needle_in_haystack": e4,
        "E5_ablations": e5,
        "E6_power_check": e6,
    }


def _sanitize(obj):
    if isinstance(obj, float):
        import math
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def compute_all_json(state: dict, settings: dict, stream: list, ground_truth: list,
                     scorer, decay_engine, retriever, comparison_builder, fact_tokens=None,
                     embed_fn=None, embedding_model="nomic-embed-text", injection_token_limit=0) -> dict:
    return _sanitize(compute_all(state, settings, stream, ground_truth,
                                 scorer, decay_engine, retriever, comparison_builder, fact_tokens,
                                 embed_fn=embed_fn, embedding_model=embedding_model,
                                 injection_token_limit=injection_token_limit))


def _baseline_window_of(state: dict, settings: dict):
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


def _per_turn_baseline_tokens(history: list, budget: int) -> list:
    result = []
    j = 0
    used = 0
    for i, h in enumerate(history):
        used += h["tokens"]
        while used > budget and j <= i:
            used -= history[j]["tokens"]
            j += 1
        result.append(used)
    return result