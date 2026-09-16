import numpy as np
from statsmodels.stats.power import TTestIndPower
from typing import Dict


def run_e6_power_check() -> Dict:
    analysis = TTestIndPower()
    effect_size = 0.5
    alpha = 0.05
    power = 0.80
    required_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)

    return {
        "target_effect_size_cohens_d": effect_size,
        "alpha": alpha,
        "target_power": power,
        "required_sample_size_per_group": int(np.ceil(required_n))
    }
