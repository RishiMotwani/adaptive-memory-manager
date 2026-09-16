import numpy as np
from typing import Dict, List


class ImportanceScorer:
    """Computes multi-factor importance scores for candidate memories."""

    def __init__(self, weights: Dict[str, float] = None):
        self.w = weights or {
            "w1_relevance": 0.40,
            "w2_utility": 0.30,
            "w3_recency": 0.15,
            "w4_frequency": 0.15
        }

    def compute_score(self, memory: Dict, query_relevance: float, current_turn: int) -> float:
        recency = 1.0 / (1.0 + 0.05 * (current_turn - memory.get("source_turn_id", current_turn)))
        frequency = min(1.0, memory.get("access_count", 1) / 10.0)
        utility = memory.get("confidence", 0.8)

        score = (
            self.w["w1_relevance"] * query_relevance +
            self.w["w2_utility"] * utility +
            self.w["w3_recency"] * recency +
            self.w["w4_frequency"] * frequency
        )
        return float(np.clip(score, 0.0, 1.0))
