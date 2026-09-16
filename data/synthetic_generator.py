import json
import random
from typing import Dict, List, Tuple


class SyntheticConversationGenerator:
    """Generates controlled long-horizon conversations with ground-truth needle facts."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.categories = ["technical_preference", "personal", "project_context", "transient"]

    def generate_conversation(self, num_turns: int = 200, signal_density: float = 0.05) -> Tuple[List[Dict], List[Dict]]:
        turns = []
        ground_truth_facts = []

        pool = list(range(10, max(1, num_turns - 20)))
        n_sig = int(num_turns * signal_density)
        if n_sig and pool:
            signal_turns = set(random.sample(pool, min(n_sig, len(pool))))
        else:
            signal_turns = set()

        for turn_id in range(1, num_turns + 1):
            if turn_id in signal_turns:
                fact_type = random.choice(["trap_old_relevant", "standard_signal", "transient_noise"])
                if fact_type == "trap_old_relevant":
                    fact_str = f"User core database requirement: primary DB must be PostgreSQL port {5432 + turn_id}"
                    cat = "technical_preference"
                    is_trap = True
                elif fact_type == "standard_signal":
                    fact_str = f"Project feature flag configuration key_{turn_id} is enabled"
                    cat = "project_context"
                    is_trap = False
                else:
                    fact_str = f"User is currently drinking coffee cup number {turn_id}"
                    cat = "transient"
                    is_trap = False

                user_msg = f"Please note this down: {fact_str}."
                gt = {
                    "source_turn": turn_id,
                    "fact": fact_str,
                    "category": cat,
                    "is_trap": is_trap,
                    "expected_needed_later": is_trap or (cat != "transient")
                }
                ground_truth_facts.append(gt)
            else:
                user_msg = f"Turn {turn_id}: Discussing routine system log outputs and generic code refactoring."

            turns.append({
                "turn_id": turn_id,
                "user": user_msg,
                "assistant": f"Acknowledged turn {turn_id} context."
            })

        return turns, ground_truth_facts


if __name__ == "__main__":
    gen = SyntheticConversationGenerator()
    convs, gts = gen.generate_conversation(50)
    print(f"Generated {len(convs)} turns with {len(gts)} embedded ground-truth memory facts.")
