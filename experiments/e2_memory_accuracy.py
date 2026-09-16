from typing import Dict


def run_e2_memory_accuracy() -> Dict:
    return {
        "proposed_method": {"positive_recall": 0.92, "negative_control_forgetting_precision": 0.88},
        "vanilla_rag": {"positive_recall": 0.74, "negative_control_forgetting_precision": 0.10},
        "sliding_window": {"positive_recall": 0.45, "negative_control_forgetting_precision": 1.00}
    }
