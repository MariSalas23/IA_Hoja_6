from __future__ import annotations

from solution.policy import Policy
from solution.lake_mdp import DOWN, RIGHT, Action, State

class RandomPolicy(Policy):
    def _decision(self, s: State) -> Action:
        # Obtiene acciones legales
        legales = list(self.mdp.actions(s))
        if not legales:
            raise ValueError("No legal actions available for state: %r" % (s,))

        # Escoge índice aleatorio
        i = self.rng.integers(len(legales))
        return legales[i]

class CustomPolicy(Policy):
    def _decision(self, s: State) -> Action:
        # Acciones legales del estado
        legales = list(self.mdp.actions(s))
        if len(legales) == 1:
            # Si solo hay ⊥ devolverla
            return legales[0]

        try:
            # Prefiere ir hacia abajo si no es hoyo
            abajo = self.mdp.intended_next_state(s, DOWN)
            if not self.mdp.is_hole(abajo):
                return DOWN
        except AttributeError:
            # Si el MDP no expone helper usar primera acción
            return legales[0]

        # Si abajo no sirve, intenta derecha
        derecha = self.mdp.intended_next_state(s, RIGHT)
        if not self.mdp.is_hole(derecha):
            return RIGHT

        # Fallback a la primera legal
        return legales[0]