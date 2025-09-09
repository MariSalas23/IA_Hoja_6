from __future__ import annotations
from typing import Iterable, List, Tuple, Dict

from solution.mdp import MDP, State, Action

UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"

class LakeMDP(MDP):
    def __init__(self, grid: Iterable[Iterable[str]]):  # Normaliza la grilla
        # Copia el grid como lista de listas
        self.grid = [list(fila) for fila in grid]
        if not self.grid or not self.grid[0]:
            raise ValueError("Grid must be a non-empty matrix.")

        # Tamaños de la grilla
        self.nfilas = len(self.grid)
        self.ncols = len(self.grid[0])

        # Verifica que todas las filas tengan la misma longitud
        for r in range(self.nfilas):
            if len(self.grid[r]) != self.ncols:
                raise ValueError("All grid rows must have the same length.")

        # Ubicaciones clave
        self._ini = None
        self._meta = None
        self._pozos = set()

        # Busca S, G y H
        for r in range(self.nfilas):
            for c in range(self.ncols):
                ch = self.grid[r][c]
                if ch == 'S':
                    if self._ini is not None:
                        raise ValueError("There must be exactly one start 'S'.")
                    self._ini = (r, c)
                elif ch == 'G':
                    if self._meta is not None:
                        raise ValueError("There must be exactly one goal 'G'.")
                    self._meta = (r, c)
                elif ch == 'H':
                    self._pozos.add((r, c))
                elif ch in ('F',):
                    # Celda transitable
                    pass
                else:
                    raise ValueError(f"Unknown cell type '{ch}' at {(r, c)}")

        if self._ini is None:
            raise ValueError("Missing start 'S' in grid.")
        if self._meta is None:
            raise ValueError("Missing goal 'G' in grid.")

    # ---------------------------- Interfaz MDP pública ----------------------------

    def start_state(self) -> State:
        # Retorna el estado inicial con etiqueta S
        r, c = self._ini
        return ((r, c), 'S')

    def actions(self, s: State) -> Iterable[Action]:
        # Devuelve acciones válidas o ⊥ si es absorbente
        if self.is_absorbed(s):
            return (ABSORB,)
        if self._es_celda(s, 'H') or self._es_celda(s, 'G'):
            return (ABSORB,)
        return (UP, RIGHT, DOWN, LEFT)

    def reward(self, s: State) -> float:
        # Recompensa según tipo de celda
        if self.is_absorbed(s):
            return 0.0
        if self._es_celda(s, 'S'):
            return 0.0
        if self._es_celda(s, 'F'):
            return 0.1
        if self._es_celda(s, 'H'):
            return -1.0
        if self._es_celda(s, 'G'):
            return 1.0
        return 0.0

    def is_terminal(self, s: State) -> bool:
        # Terminal si está absorbido
        return self.is_absorbed(s)

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
        # Transición estocástica con slippage lateral
        if self.is_absorbed(s):
            return [(self.absorb_state(), 1.0)]
        if self._es_celda(s, 'H') or self._es_celda(s, 'G'):
            return [(self.absorb_state(), 1.0)]
        if a not in (UP, RIGHT, DOWN, LEFT):
            # Acción inválida permanece en el estado
            return [(s, 1.0)]

        principal = a
        lat_izq, lat_der = self._laterales(a)
        probas = ((principal, 0.8), (lat_izq, 0.1), (lat_der, 0.1))

        acumulado: Dict[State, float] = {}
        for act, p in probas:
            ns = self.intended_next_state(s, act)
            # Acumula probabilidades en el mismo estado destino
            acumulado[ns] = acumulado.get(ns, 0.0) + p

        return list(acumulado.items())

    # --------------------------------- Helpers ---------------------------------

    def absorb_state(self) -> State:
        # Estado absorbente canónico
        return (ABSORB, ABSORB)

    def is_absorbed(self, s: State) -> bool:
        # Verdadero si es el par (⊥, ⊥)
        return isinstance(s, tuple) and len(s) == 2 and s[0] == ABSORB and s[1] == ABSORB

    def _en_rango(self, r: int, c: int) -> bool:
        # Verifica que esté dentro del tablero
        return 0 <= r < self.nfilas and 0 <= c < self.ncols

    def _laterales(self, a: Action):
        # Retorna acciones laterales según la principal
        if a in (UP, DOWN):
            return (LEFT, RIGHT)
        if a in (LEFT, RIGHT):
            return (UP, DOWN)
        return (UP, DOWN)

    def is_goal(self, s: State) -> bool:
        # Indica si es meta
        return self._es_celda(s, 'G')

    def is_cell(self, s: State) -> str:
        # Devuelve letra de la celda actual
        return self._tipo_celda(s)

    def intended_next_state(self, s: State, a: Action) -> State:
        # Movimiento determinista basado en la acción
        if self.is_absorbed(s):
            return self.absorb_state()

        if isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple) and isinstance(s[1], str):
            (r, c), _ = s
        else:
            # Soporta formato alterno
            r, c = s

        dr, dc = 0, 0
        if a == UP:
            dr, dc = -1, 0
        elif a == RIGHT:
            dr, dc = 0, 1
        elif a == DOWN:
            dr, dc = 1, 0
        elif a == LEFT:
            dr, dc = 0, -1
        else:
            return s

        nr, nc = r + dr, c + dc
        if not self._en_rango(nr, nc):
            # Choque con borde mantiene estado
            return s

        return ((nr, nc), self.grid[nr][nc])

    def _tipo_celda(self, rc: State) -> str:
        # Devuelve símbolo de celda o ⊥
        if self.is_absorbed(rc):
            return ABSORB

        if isinstance(rc, tuple) and len(rc) == 2 and isinstance(rc[0], tuple) and isinstance(rc[1], str):
            return rc[1]

        r, c = rc
        return self.grid[r][c]

    def _es_celda(self, s: State, ch: str) -> bool:
        # Compara la celda con un símbolo
        return self._tipo_celda(s) == ch

    def is_hole(self, s: State) -> bool:
        # Indica si es un hoyo
        return self._es_celda(s, 'H')