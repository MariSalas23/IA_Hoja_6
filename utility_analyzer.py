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
        rng = np.random.default_rng(int(seed))
        policy = policy_cls(self.mdp, rng)

        s = self.mdp.start_state()
        utility = 0.0
        t = 0
        terminal_kind: TerminalKind = "none"

        while t < self.step_limit:
            actions = list(self.mdp.actions(s))
            if actions == ['⊥'] or (len(actions) == 1 and actions[0] == '⊥'):  
                break

            a: Action = policy(s)
            ns, r = self.mdp.step(s, a, rng)
            utility += (self.gamma ** t) * r

            entered_terminal = (
                isinstance(ns, tuple)
                and self.mdp.reward(ns) in (-1.0, 1.0)
            )
            if entered_terminal:
                terminal_kind = "goal" if r > 0 else "hole"
                t += 1
                s = ns
                break

            s = ns
            t += 1

        length = t
        
        return float(utility), int(length), terminal_kind

    def evaluate(
        self, policy_cls: Type[Policy], n_trials: int, base_seed: int = 0
    ) -> Dict[str, Any]:
        utils = []
        lengths = []
        counts = {"goal": 0, "hole": 0, "none": 0}

        n_trials = int(n_trials)
        base_seed = int(base_seed)

        for i in range(n_trials):
            u, L, kind = self.run_trial(policy_cls, seed=base_seed + i)
            utils.append(u)
            lengths.append(L)
            counts[kind] += 1

        n = max(1, n_trials)
        mean_util = float(np.mean(utils)) if utils else 0.0
        var_util = float(np.var(utils)) if utils else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0

        return {
            "n_trials": n_trials,
            "mean_utility": mean_util,
            "utility_variance": var_util,
            "p_goal": counts["goal"] / n,
            "p_hole": counts["hole"] / n,
            "p_none": counts["none"] / n,
            "mean_length": mean_len,
        }