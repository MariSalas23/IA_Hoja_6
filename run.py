from __future__ import annotations
import json

from solution.lake_mdp import LakeMDP
from solution.policies import RandomPolicy, CustomPolicy
from solution.utility_analyzer import UtilityAnalyzer

DEFAULT_MAP = ['SFFF', 'FHFH', 'FFFH', 'HFFG']

def evaluate_all(trials: int = 100, base_seed: int = 123):
    """
    Evaluate RandomPolicy and CustomPolicy for γ ∈ {0.5, 0.9, 1.0}.
    Returns a JSON-serializable dict with summaries and the winner per γ.
    """
    report = {"n_trials": int(trials), "base_seed": int(base_seed), "gammas": {}}

    for gamma in (0.5, 0.9, 1.0):
        mdp = LakeMDP(DEFAULT_MAP)
        ua = UtilityAnalyzer(mdp=mdp, gamma=float(gamma), step_limit=100)

        sum_rand = ua.evaluate(RandomPolicy, n_trials=int(trials), base_seed=int(base_seed))
        sum_cust = ua.evaluate(CustomPolicy, n_trials=int(trials), base_seed=int(base_seed) + 10_000)

        mu_r = sum_rand.get("mean_utility", 0.0)
        mu_c = sum_cust.get("mean_utility", 0.0)

        if abs(mu_r - mu_c) > 1e-12:
            best = "custom" if mu_c > mu_r else "random"
        else:
            vr = sum_rand.get("utility_variance", 0.0)
            vc = sum_cust.get("utility_variance", 0.0)
            if vc < vr:
                best = "custom"
            elif vr < vc:
                best = "random"
            else:
                best = "tie"

        report["gammas"][str(gamma)] = {
            "random": sum_rand,
            "custom": sum_cust,
            "winner": best,
        }

    return report