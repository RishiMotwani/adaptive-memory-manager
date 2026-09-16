"""Retrieval over the adaptive memory store (brain.md D3/D6).

The original docstring claimed "vector embedding cosine similarity search" but
the implementation was lexical token overlap only. Now the primary path really is
dense retrieval via Ollama embeddings (nomic-embed-text, 768-dim): candidates
are scored by cosine similarity and re-ranked by a blend of importance and
similarity plus a small category-match bonus. The lexical fallback is kept solely
for when the embedding endpoint is unavailable, and each returned memory records
which engine produced it (retrieval_engine) so E5 stays comparable.

Reinforcement (brain.md D4): memories actually *selected* for injection have
their access_count bumped and last_access_turn refreshed, feeding the decay
engine's usage-fed importance (SF-AMS / Oblivion / MemoryBank).
"""

import time
from typing import Callable, Dict, List, Optional

from memory_optimizer.budget import fit_to_budget
from memory_optimizer.extraction import _keyword_category

FactTokens = Callable[[str], int]


def _token_overlap(query: str, fact: str) -> float:
    q_words = set(query.lower().split())
    m_words = set(fact.lower().split())
    exact = q_words.intersection(m_words)
    if not exact:
        exact = {q for q in q_words for m in m_words if q in m or m in q}
    return len(exact) / (max(len(q_words), 1))


def _cosine(a, b):
    if a is None or b is None:
        return None
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


class MemoryRetriever:
    """Real embedding cosine similarity search over stored memory items."""

    def __init__(
        self,
        top_k: int = 5,
        sim_threshold: float = 0.35,
        embed_fn: Callable[[List[str]], list] = None,
        embedding_model: str = "nomic-embed-text",
        imp_weight: float = 0.6,
        sim_weight: float = 0.4,
        cat_bonus: float = 0.1,
    ):
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.embed_fn = embed_fn
        self.embedding_model = embedding_model
        self.imp_weight = imp_weight
        self.sim_weight = sim_weight
        self.cat_bonus = cat_bonus
        self.stats = {"embed_calls": 0, "embed_ms": 0.0}

    def _embed(self, text: str) -> Optional[list]:
        """Embeds a single text (through the shared cache) and times it."""
        if self.embed_fn is None:
            return None
        t0 = time.perf_counter()
        try:
            vec = self.embed_fn([text])
            vec = vec[0] if isinstance(vec, (list, tuple)) and len(vec) == 1 else vec
        except Exception:
            vec = None
        self.stats["embed_ms"] += (time.perf_counter() - t0) * 1000
        self.stats["embed_calls"] += 1
        return vec

    def _ensure_embedding(self, mem: Dict, text: str) -> None:
        if "fact_embedding" not in mem and self.embed_fn is not None:
            mem["fact_embedding"] = self._embed(text)

    def _bump(self, mem: Dict, current_turn: Optional[int]) -> None:
        if current_turn is None:
            return
        mem["access_count"] = int(mem.get("access_count", 1)) + 1
        mem["last_access_turn"] = current_turn

    def retrieve(
        self,
        query: str,
        memories: List[Dict],
        current_turn: Optional[int] = None,
        token_limit: int = 0,
        fact_tokens: FactTokens = None,
    ) -> List[Dict]:
        self.stats = {"embed_calls": 0, "embed_ms": 0.0}

        qvec = self._embed(query)
        scored: List[Dict] = []
        for mem in memories:
            self._ensure_embedding(mem, mem["fact"])
            importance = mem.get("current_importance", mem.get("base_score", 0))

            sim = None
            engine = "lexical"
            if qvec is not None and mem.get("fact_embedding") is not None:
                sim = _cosine(qvec, mem["fact_embedding"])
                engine = "embedding"
            else:
                sim = _token_overlap(query, mem["fact"])
                engine = "lexical"
            if sim is None:
                sim = 0.0

            if sim >= self.sim_threshold or importance > 0.6:
                qcat = _keyword_category(query)
                bonus = self.cat_bonus if (qcat and mem.get("category") == qcat) else 0.0
                copy = mem.copy()
                copy["retrieval_sim"] = round(sim, 4)
                copy["importance"] = importance
                copy["retrieval_engine"] = engine
                copy["retrieval_bonus"] = bonus
                copy["_source"] = mem
                scored.append(copy)

        scored.sort(
            key=lambda x: (
                self.imp_weight * x["importance"]
                + self.sim_weight * (x["retrieval_sim"] + x["retrieval_bonus"])
            ),
            reverse=True,
        )

        if token_limit and token_limit > 0 and fact_tokens is not None:
            selected, _, _ = fit_to_budget(scored, token_limit, fact_tokens)
        else:
            selected = scored[: self.top_k]

        for item in selected:
            src = item.get("_source")
            if src is not None:
                self._bump(src, current_turn)
                item["_source"] = None
                item.pop("_source", None)

        return [dict(i) for i in selected]