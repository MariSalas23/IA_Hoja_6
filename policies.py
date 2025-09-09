from __future__ import annotations

from solution.policy import Policy
from solution.lake_mdp import DOWN, RIGHT, Action, State
class RandomPolicy(Policy):
    """Uniform over legal actions."""

    def _decision(self, s: State) -> Action:
        actions = list(self.mdp.actions(s))
        if not actions:
            
            return '⊥'
        
        idx = self.rng.integers(0, len(actions))

        return actions[idx]
class CustomPolicy(Policy):
    """
    Simple deterministic rule that avoids an immediate hole:
      - Prefer DOWN if the cell below is NOT a hole.
      - Else prefer RIGHT if the cell to the right is NOT a hole.
      - Else pick the first legal action (covers absorbing ⊥ case, or being boxed in).
    Note: This checks intended cells only (not slip outcomes).
    """

    def _decision(self, s: State) -> Action:
        actions = list(self.mdp.actions(s))
        if not actions:
            
            return '⊥'

        def neighbor(state: State, act: Action) -> State:
            if not isinstance(state, tuple):
               
                return state
            
            r, c = state
            if act == 'UP':
                nr, nc = r - 1, c
            elif act == 'DOWN':
                nr, nc = r + 1, c
            elif act == 'LEFT':
                nr, nc = r, c - 1
            elif act == 'RIGHT':
                nr, nc = r, c + 1
            else:
                
                return state
           
            if nr < 0 or nc < 0:
                
                return (r, c)
            
            return (nr, nc)

        def is_hole_cell(cell: State) -> bool:
   
            try:
                
                return self.mdp.reward(cell) < 0
            
            except Exception:
                
                return False

        if DOWN in actions:
            nb = neighbor(s, DOWN)
            if not is_hole_cell(nb):
                
                return DOWN

        if RIGHT in actions:
            nb = neighbor(s, RIGHT)
            if not is_hole_cell(nb):
                
                return RIGHT

        return actions[0]