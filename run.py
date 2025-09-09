from __future__ import annotations
import json

from solution.lake_mdp import LakeMDP
from solution.policies import RandomPolicy, CustomPolicy
from solution.utility_analyzer import UtilityAnalyzer

# Default map (4x4 from the assignment PDF)
DEFAULT_MAP = # TODO: implement

def evaluate_all(trials: int = 100, base_seed: int = 123):
    """
    Evaluate RandomPolicy and CustomPolicy for γ ∈ {0.5, 0.9, 1.0}.
    Returns a JSON-serializable dict with summaries and the winner per γ.
    """
    report = {"n_trials": int(trials), "base_seed": int(base_seed), "gammas": {}}

    # TODO: implement


    report["gammas"][str(gamma)] = {
        "random": sum_rand,
        "custom": sum_cust,
        "winner": best,
    }

    return report