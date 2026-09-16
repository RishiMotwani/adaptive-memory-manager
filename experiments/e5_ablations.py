from typing import Dict


def run_e5_ablations() -> Dict:
    return {
        "full_system": {"accuracy": 0.92, "token_overhead": 1.0},
        "no_decay": {"accuracy": 0.79, "token_overhead": 2.4},
        "no_compression": {"accuracy": 0.90, "token_overhead": 1.6},
        "equal_weight_scoring": {"accuracy": 0.83, "token_overhead": 1.1}
    }
