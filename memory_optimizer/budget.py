"""Shared token-budget logic (brain.md D1/D2/D7).

Both the live pipeline (server._ingest_turn) and the offline E5 replay must apply
the *same* budget semantics, so this module is the single source of truth:

- token_budget_evict: enforce a hard token cap on the adaptive store by evicting
  the lowest-current-importance memories (oldest source turn as tie-break),
  never overshooting the budget. Evidence: SF-AMS utility-driven survival
  (arXiv:2607.22562), MemoryBank fade curves (AAAI 2024), MemGPT
  evict-under-pressure (arXiv:2310.08560).
- fit_to_budget: greedy token-aware injection fitting (brain.md D7 fill mode) —
  ranked candidates are selected while they fit, filling the rest of the limit
  with the next-best by rank instead of stopping at top_k.
"""

from typing import Callable, Dict, List, Tuple

FactTokens = Callable[[str], int]


def _tokens(mem: Dict, fact_tokens: FactTokens) -> int:
    if fact_tokens is not None:
        v = fact_tokens(mem.get("fact", ""))
        if v is not None:
            return int(v)
    v = mem.get("measured_tokens")
    return int(v) if v is not None else 0


def _sum_tokens(memories: List[Dict], fact_tokens: FactTokens) -> int:
    return sum(_tokens(m, fact_tokens) for m in memories)


def token_budget_evict(
    memories: List[Dict],
    budget: int,
    fact_tokens: FactTokens = None,
    importance_key: str = "current_importance",
) -> Tuple[List[Dict], List[Dict], int]:
    """Evict lowest-importance memories until the active store fits `budget`.

    Returns (kept, evicted, used_tokens). Eviction keys on the memory's current
    importance, breaking ties toward the oldest source turn so recent,
    reinforced memories survive the squeeze.
    """
    kept = [m for m in memories]
    used = _sum_tokens(kept, fact_tokens)
    evicted: List[Dict] = []
    while used > budget and kept:
        kept.sort(
            key=lambda m: (
                m.get(importance_key, m.get("base_score", 0)),
                m.get("source_turn_id", 0),
            )
        )
        victim = kept.pop(0)
        evicted.append(victim)
        used -= _tokens(victim, fact_tokens)
    return kept, evicted, used


def fit_to_budget(
    ranked: List[Dict],
    token_limit: int,
    fact_tokens: FactTokens = None,
) -> Tuple[List[Dict], List[Dict], int]:
    """Greedy token-aware selection of injected facts (D7 fill mode).

    Selects ranked candidates while their measured sizes fit `token_limit`.
    Never returns empty when candidates exist: if even the top candidate alone
    overshoots the limit, it is still injected (min-1 guarantee) so the answer
    path always has context. Returns (selected, skipped, used_tokens).
    """
    selected: List[Dict] = []
    skipped: List[Dict] = []
    used = 0
    for m in ranked:
        size = _tokens(m, fact_tokens)
        if used + size <= token_limit:
            selected.append(m)
            used += size
        else:
            skipped.append(m)
    if not selected and ranked:
        top = ranked[0]
        size = _tokens(top, fact_tokens) or 1
        selected = [top]
        used = size
        skipped = ranked[1:]
    return selected, skipped, used