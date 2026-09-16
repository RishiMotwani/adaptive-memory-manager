import json
import os
import sys
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from experiments.e1_token_efficiency import run_e1_token_efficiency
from experiments.e2_memory_accuracy import run_e2_memory_accuracy
from experiments.e3_latency import run_e3_latency
from experiments.e4_needle_in_haystack import run_e4_needle_in_haystack
from experiments.e5_ablations import run_e5_ablations
from experiments.e6_power_check import run_e6_power_check


def main():
    print("=========================================================")
    print(" RUNNING ADAPTIVE MEMORY MANAGER EXPERIMENTAL SUITE")
    print("=========================================================")

    with open(BASE_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results = {
        "config": config,
        "E1_token_efficiency": run_e1_token_efficiency(),
        "E2_memory_accuracy": run_e2_memory_accuracy(),
        "E3_latency": run_e3_latency(),
        "E4_needle_in_haystack": run_e4_needle_in_haystack(),
        "E5_ablations": run_e5_ablations(),
        "E6_power_check": run_e6_power_check()
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/experiment_manifest.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] All experiments executed successfully.")
    print(f"[MANIFEST] Full statistical log saved to: {out_path}\n")


if __name__ == "__main__":
    main()
