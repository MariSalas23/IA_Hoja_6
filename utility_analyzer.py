from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Any, Type
import numpy as np

from solution.mdp import MDP, Action
from solution.policies import Policy

TerminalKind = Literal["goal", "hole", "none"]


@dataclass
class UtilityAnalyzer:
    mdp: MDP
    gamma: float = 0.99
    step_limit: int = 100

    def run_trial(
        self, policy_cls: Type[Policy], seed: int
    ) -> Tuple[float, int, TerminalKind]:
        """
        Instantiate a fresh policy with its own rng(seed) and simulate one episode.
        Returns (discounted_utility, length, terminal_kind).
        """
        # TODO: implement

    def evaluate(
        self, policy_cls: Type[Policy], n_trials: int, base_seed: int = 0
    ) -> Dict[str, Any]:
        # TODO: implement

        return {
            "n_trials": # n_trials,
            "mean_utility": # mean_util
            "utility_variance": # var_util
            "p_goal": # probability of reaching goal,
            "p_hole": # probability of falling in hole,
            "p_none": # probability of hitting step limit,
            "mean_length": #  mean run length
        }
