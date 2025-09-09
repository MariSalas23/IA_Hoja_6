from __future__ import annotations
import json

from solution.lake_mdp import LakeMDP
from solution.policies import RandomPolicy, CustomPolicy
from solution.utility_analyzer import UtilityAnalyzer

DEFAULT_MAP = [  # Mapa por defecto
    ['S', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'F'],
    ['F', 'F', 'F', 'F'],
    ['H', 'F', 'F', 'G'],
]

def evaluate_all(trials: int = 100, base_seed: int = 123):
    # Prepara el reporte base
    resumen = {"n_trials": int(trials), "base_seed": int(base_seed), "gammas": {}}

    for g in (0.5, 0.9, 1.0):
        # Crea el MDP y el analizador
        problema = LakeMDP(DEFAULT_MAP)
        analisis = UtilityAnalyzer(problema, gamma=float(g))

        # Evalúa políticas con la misma configuración de semillas
        stats_rand = analisis.evaluate(RandomPolicy, int(trials), base_seed=int(base_seed))
        stats_custom = analisis.evaluate(CustomPolicy, int(trials), base_seed=int(base_seed))

        ur, uc = stats_rand["mean_utility"], stats_custom["mean_utility"]

        # Selecciona ganador con desempates por varianza y prob de llegar a meta
        if uc > ur:
            ganador = "custom"
        elif ur > uc:
            ganador = "random"
        else:
            vr, vc = stats_rand["utility_variance"], stats_custom["utility_variance"]
            if vc < vr:
                ganador = "custom"
            elif vr < vc:
                ganador = "random"
            else:
                ganador = "custom" if stats_custom.get("p_goal", 0.0) >= stats_rand.get("p_goal", 0.0) else "random"

        resumen["gammas"][str(g)] = {
            "random": stats_rand,
            "custom": stats_custom,
            "winner": ganador,
        }

    return resumen

if __name__ == "__main__":
    # Imprime el reporte en JSON
    print(json.dumps(evaluate_all(), indent=2))