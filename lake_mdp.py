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

    def __init__(self, grid: Iterable[Iterable[str]]): # Normalizar
        self._grid: List[List[str]] = [list(row) for row in grid]
        self._rows = len(self._grid)
        if self._rows == 0:
            raise ValueError("Empty grid.")
        self._cols = len(self._grid[0])
        for row in self._grid:
            if len(row) != self._cols:
                raise ValueError("Jagged grid.")

        starts = [(r, c) # Inicio
                  for r in range(self._rows)
                  for c in range(self._cols)
                  if self._grid[r][c] == 'S']
        if len(starts) != 1:
            raise ValueError("Grid must contain exactly one 'S' start cell.")
        self._start: Tuple[int, int] = starts[0]

        self._terminals = {'H', 'G'} # Precompute terminal H y G

        self._slip_probs = (0.8, 0.1, 0.1) # Movimientos

    # --- MDP interface -----------------------------------------------------
    def start_state(self) -> State:

        return self._start

    def actions(self, s: State) -> Iterable[Action]:
        if s == '⊥': # ⊥ solo tiene acción ⊥
            return (ABSORB,)
        # If we are currently standing on H/G, only ⊥ is legal.
        if isinstance(s, tuple):
            r, c = s
            if self._grid[r][c] in self._terminals:
                return (ABSORB,)

        return (UP, RIGHT, DOWN, LEFT) # Movimientos válidos

    def reward(self, s: State) -> float:
        if s == '⊥':
            return 0.0
        if not isinstance(s, tuple):
            return 0.0
        r, c = s
        cell = self._grid[r][c]
        if cell == 'F':
            return 0.1
        if cell == 'H':
            return -1.0
        if cell == 'G':
            return 1.0
        if cell == 'S':
 
            return 0.0 # No recompensa por entrar S

        return 0.0

    def is_terminal(self, s: State) -> bool:

        return s == '⊥'

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
  
        if s == '⊥': # Si ya está absorbiendo
            return [('⊥', 1.0)]

        if isinstance(s, tuple):
            r, c = s
            cell = self._grid[r][c]
            if cell in self._terminals:

                return [('⊥', 1.0)]

        if a == ABSORB:

            return [(s, 1.0)]

        intended, left_lateral, right_lateral = self._lateral_actions(a)
        probs = self._slip_probs
        moves = (intended, left_lateral, right_lateral)

        outcomes: Dict[State, float] = {}
        for mv, p in zip(moves, probs):
            ns = self._move_with_bump(s, mv)
            outcomes[ns] = outcomes.get(ns, 0.0) + p

        total = sum(outcomes.values())
        if total <= 0:

            return [(s, 1.0)]
        
        return [(ns, p / total) for ns, p in outcomes.items()]

    # --- helpers -----------------------------------------------------------
    def _move_with_bump(self, s: State, a: Action) -> State:
        if not isinstance(s, tuple):

            return s
        
        r, c = s
        dr, dc = 0, 0
        if a == UP:
            dr, dc = -1, 0
        elif a == DOWN:
            dr, dc = 1, 0
        elif a == LEFT:
            dr, dc = 0, -1
        elif a == RIGHT:
            dr, dc = 0, 1
        nr, nc = r + dr, c + dc
        if 0 <= nr < self._rows and 0 <= nc < self._cols:
            
            return (nr, nc)

        return (r, c)

    def _lateral_actions(self, a: Action) -> Tuple[Action, Action, Action]:
       
        if a == UP:
            return (UP, LEFT, RIGHT)
        if a == DOWN:
            return (DOWN, RIGHT, LEFT)
        if a == LEFT:
            return (LEFT, DOWN, UP)
        if a == RIGHT:
            return (RIGHT, UP, DOWN)
        
        return (a, a, a)  # Returna intended, lateral_left, lateral_right