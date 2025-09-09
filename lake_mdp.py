from __future__ import annotations
from typing import Iterable, List, Tuple, Dict

from solution.mdp import MDP, State, Action

UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"


class LakeMDP(MDP):
    """
    Grid map (matrix of single-character strings), e.g.:
      [
        ['S','F','F','F'],
        ['F','H','F','F'],
        ['F','F','F','F'],
        ['H','F','F','G'],
      ]

    Rewards are *state-entry* rewards. After entering H or G, the next state is
    the absorbing state ⊥ with only legal action ⊥ and 0 reward forever.
    """

    def __init__(self, grid: Iterable[Iterable[str]]):
        # TODO: implement

    # --- MDP interface -----------------------------------------------------
    def start_state(self) -> State:
        # TODO: implement

    def actions(self, s: State) -> Iterable[Action]:
        # TODO: implement

    def reward(self, s: State) -> float:
        # TODO: implement

    def is_terminal(self, s: State) -> bool:
        # TODO: implement

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
        # TODO: implement

    # --- helpers -----------------------------------------------------------
    # TODO: implement any helper methods you need