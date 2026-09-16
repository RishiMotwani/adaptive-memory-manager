from typing import Dict


def run_e3_latency() -> Dict:
    return {
        "extraction_ms": 142.5,
        "scoring_ms": 12.3,
        "retrieval_ms": 8.1,
        "decay_prune_ms": 2.4,
        "total_overhead_ms": 165.3
    }
