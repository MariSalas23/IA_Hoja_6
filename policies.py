from __future__ import annotations

from solution.policy import Policy
from solution.lake_mdp import DOWN, RIGHT, Action, State


class RandomPolicy(Policy):
    """Uniform over legal actions."""

    def _decision(self, s: State) -> Action:
        # TODO: implement


class CustomPolicy(Policy):
    """
    Simple deterministic rule that avoids an immediate hole:
      - Prefer DOWN if the cell below is NOT a hole.
      - Else prefer RIGHT if the cell to the right is NOT a hole.
      - Else pick the first legal action (covers absorbing ⊥ case, or being boxed in).
    Note: This checks intended cells only (not slip outcomes).
    """

    def _decision(self, s: State) -> Action:
        # TODO: implement
