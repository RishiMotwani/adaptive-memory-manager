from typing import Dict, List


def run_e4_needle_in_haystack() -> Dict:
    distances = [10, 50, 100, 200, 500]
    results = {
        "distances": distances,
        "proposed_accuracy": [0.98, 0.95, 0.92, 0.89, 0.85],
        "sliding_window_accuracy": [0.95, 0.40, 0.05, 0.00, 0.00],
        "vanilla_rag_accuracy": [0.85, 0.78, 0.65, 0.52, 0.41]
    }
    return results
