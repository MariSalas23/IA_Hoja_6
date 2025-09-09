from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Any, Type
import numpy as np

from solution.mdp import MDP, Action
from solution.policies import Policy

# Tipos de terminación
TerminalKind = Literal["goal", "hole", "none"]

@dataclass
class UtilityAnalyzer:
    mdp: MDP
    gamma: float = 0.99
    step_limit: int = 100  # Límite para evitar ciclos

    def run_trial(
        self, policy_cls: Type[Policy], seed: int
    ) -> Tuple[float, int, TerminalKind]:
        # Inicializa generadores aleatorios
        alea_pol = np.random.default_rng(seed)
        alea_env = np.random.default_rng(seed + 1)

        # Crea la política con el MDP
        policy = policy_cls(self.mdp, alea_pol)

        st = self.mdp.start_state()  # Estado inicial
        ret = 0.0                    # Retorno acumulado
        df = 1.0                     # Factor de descuento
        pasos = 0                    # Longitud del episodio
        final: TerminalKind = "none" # Marca de terminación

        # Ejecuta hasta step_limit o hasta absorber
        while pasos < self.step_limit:
            # Selecciona acción de la política
            act: Action = policy(st)

            # Avanza en el MDP
            nxt, rew = self.mdp.step(st, act, alea_env)

            # Acumula retorno descontado
            ret += df * rew
            pasos += 1

            # Revisa si el siguiente estado es terminal por única acción ⊥
            acciones_sig = list(self.mdp.actions(nxt))
            if len(acciones_sig) == 1 and acciones_sig[0] == '⊥':
                # Determina el tipo de finalización según la recompensa
                if rew > 0:
                    final = "goal"
                elif rew < 0:
                    final = "hole"
                else:
                    final = "none"
                break

            # Actualiza descuento y estado
            st = nxt
            df *= self.gamma

        return float(ret), int(pasos), final

    def evaluate(
        self, policy_cls: Type[Policy], n_trials: int, base_seed: int = 0
    ) -> Dict[str, Any]:
        # Acumuladores de métricas
        lista_util = []
        lista_len = []
        tipos = {"goal": 0, "hole": 0, "none": 0}

        # Corre múltiples episodios
        for k in range(int(n_trials)):
            u, L, t = self.run_trial(policy_cls, base_seed + k)
            lista_util.append(u)
            lista_len.append(L)
            tipos[t] += 1

        # Calcula estadísticas
        arr_u = np.array(lista_util, dtype=float)
        media_u = float(arr_u.mean()) if arr_u.size else 0.0
        var_u = float(arr_u.var()) if arr_u.size else 0.0
        media_L = float(np.mean(lista_len)) if lista_len else 0.0

        return {
            "n_trials": int(n_trials),
            "mean_utility": media_u,
            "utility_variance": var_u,
            "p_goal": tipos["goal"] / float(n_trials) if n_trials else 0.0,
            "p_hole": tipos["hole"] / float(n_trials) if n_trials else 0.0,
            "p_none": tipos["none"] / float(n_trials) if n_trials else 0.0,
            "mean_length": media_L,
        }