import math
from typing import Dict, List, Tuple


class CategoryDecayEngine:
    """Usage-fed exponential forgetting (brain.md D4).

    Grounded in the forgetting curves of MemoryBank (AAAI 2024), SF-AMS strategic
    forgetting (arXiv:2607.22562) and Oblivion (EMNLP 2026): decay is keyed on
    time-since-*last-access* (not creation), and every access/re-mention
    reinforces the memory's importance with a capped gain instead of a one-shot
    base_score. M(t) = min(1, base_score * (1 + gain * (access_count - 1)))
    * exp(-lambda_c * (current_turn - last_access_turn)).
    """

    def __init__(self, lambdas: Dict[str, float] = None, pruning_threshold: float = 0.20,
                 access_gain: float = 0.2):
        self.lambdas = lambdas or {
            "transient": 0.15,
            "personal": 0.005,
            "technical_preference": 0.02,
            "project_context": 0.01,
        }
        self.pruning_threshold = pruning_threshold
        self.access_gain = access_gain

    def calculate_decayed_importance(self, memory: Dict, current_turn: int) -> float:
        cat = memory.get("category", "transient")
        lambda_c = self.lambdas.get(cat, self.lambdas.get("transient", 0.15))
        last = memory.get("last_access_turn", memory.get("source_turn_id", current_turn))
        delta_t = max(0, current_turn - last)
        base_importance = memory.get("base_score", 1.0)
        access_count = max(1, int(memory.get("access_count", 1)))
        reinforced = min(1.0, base_importance * (1.0 + self.access_gain * (access_count - 1)))
        return reinforced * math.exp(-lambda_c * delta_t)

    def step_decay_and_prune(self, memories: List[Dict], current_turn: int) -> Tuple[List[Dict], List[Dict]]:
        active = []
        pruned = []
        for mem in memories:
            m_t = self.calculate_decayed_importance(mem, current_turn)
            mem["current_importance"] = m_t
            if m_t >= self.pruning_threshold:
                active.append(mem)
            else:
                mem["prune_reason"] = f"M(t)={m_t:.3f} < threshold ({self.pruning_threshold})"
                pruned.append(mem)
        return active, pruned