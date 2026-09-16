from typing import Dict, List


def _overlap(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    exact = a_set.intersection(b_set)
    if not exact:
        exact = {x for x in a_set for y in b_set if x in y or y in x}
    return len(exact) / (max(len(a_set), 1))


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


class MemoryCompressor:
    """Deduplicates redundant atomic facts within similar clusters.

    Per brain.md D5 research (summarization drift in arXiv:2603.07670;
    "deduplicate, don't summarize" in arXiv:2605.08538; Mem0 add/update/noop in
    the VanillaRAG/Mem0 line of work): consolidation is done by *semantic
    dedupe*, not by LLM summarization. Deduping a fact counts as a re-mention:
    it reinforces the surviving memory (access_count++ and last_access_turn
    refresh) just like a retrieval would.
    """

    def compress_cluster(self, facts: List[Dict]) -> Dict:
        """Concatenative placeholder — NOT a real summarizer.

        Kept only for API compatibility. A true LLM summarizer is explicitly the
        highest-risk maneuver in the memory stack (compression drift) and this
        stub is never invoked on the live ingest path. Do not "improve" it into
        a model call; consolidate via dedupe instead.
        """
        if not facts:
            return {}
        if len(facts) == 1:
            return facts[0]

        primary = facts[0]
        combined_text = "; ".join([f["fact"] for f in facts])
        compressed_fact = primary.copy()
        compressed_fact["fact"] = f"Summarized Context: {combined_text}"
        compressed_fact["compressed_from"] = len(facts)
        return compressed_fact

    def dedupe(self, memories: List[Dict], overlap_threshold: float = 0.85) -> List[Dict]:
        """Merge near-duplicate facts of the same category, keeping the stronger one."""
        result: List[Dict] = []
        for mem in memories:
            merged = False
            for existing in result:
                if existing["category"] == mem["category"] and _overlap(existing["fact"], mem["fact"]) >= overlap_threshold:
                    existing["fact"] = existing["fact"]
                    existing["confidence"] = max(existing.get("confidence", 0.5), mem.get("confidence", 0.5))
                    existing["access_count"] = existing.get("access_count", 1) + mem.get("access_count", 1)
                    existing["last_access_turn"] = mem.get("last_access_turn", existing.get("last_access_turn"))
                    existing["duplicates"] = existing.get("duplicates", 1) + 1
                    merged = True
                    break
            if not merged:
                result.append(dict(mem))
        return result

    def dedupe_incremental(self, existing: List[Dict], new_items: List[Dict],
                           overlap_threshold: float = 0.85, embed_fn=None) -> List[Dict]:
        """Merge only newly ingested facts into the existing store (O(new x existing)).

        Same-category facts merge if their embeddings agree (cos >= 0.90) or,
        when embeddings are unavailable for the pair, if lexical overlap clears
        the threshold. The surviving fact is reinforced (re-mention semantics)
        and keeps embeddings on the merged item when one was computed.
        """
        for mem in new_items:
            merged = False
            mem_emb = None
            if embed_fn is not None:
                try:
                    mem_emb = embed_fn([mem["fact"]])
                    mem_emb = mem_emb[0] if isinstance(mem_emb, (list, tuple)) and len(mem_emb) == 1 else mem_emb
                except Exception:
                    mem_emb = None
            if mem_emb is not None:
                mem["fact_embedding"] = mem_emb
            for ex in existing:
                if ex["category"] != mem["category"]:
                    continue
                ex_emb = ex.get("fact_embedding")
                sim = None
                semantic = False
                if mem_emb is not None and ex_emb is not None:
                    cos = _cosine(mem_emb, ex_emb)
                    if cos is not None:
                        sim = cos
                        semantic = True
                else:
                    sim = _overlap(ex["fact"], mem["fact"])
                threshold = 0.90 if semantic else overlap_threshold
                if sim is not None and sim >= threshold:
                    ex["confidence"] = max(ex.get("confidence", 0.5), mem.get("confidence", 0.5))
                    ex["access_count"] = ex.get("access_count", 1) + mem.get("access_count", 1)
                    ex["last_access_turn"] = mem.get("last_access_turn", ex.get("last_access_turn",
                                                                                  ex.get("source_turn_id")))
                    ex["duplicates"] = ex.get("duplicates", 1) + 1
                    if ex_emb is None and mem_emb is not None:
                        ex["fact_embedding"] = mem_emb
                    merged = True
                    break
            if not merged:
                existing.append(dict(mem))
        return existing