"""NSGA-II (Deb et al., 2002) para el problema biobjetivo.   Responsable: Persona A

Objetivos (ambos a minimizar): fitness.multi(mask) -> (error CV, numero de variables).
Devuelve el frente de Pareto aproximado en Result.pareto como lista de
(error, n_variables, mascara).

Pseudocodigo
------------
    P <- pop_size mascaras aleatorias (con distintas densidades: p_init en [0.05, 0.5])
    evaluar objetivos de P
    mientras quede presupuesto:
        Q <- hijos por torneo binario (rango, luego crowding), cruce uniforme, mutacion
        evaluar Q
        R <- P + Q
        frentes <- ordenacion_no_dominada(R)          # frente 1, 2, 3...
        P <- llenar con frentes completos; el ultimo que no cabe entero se corta
             quedandose con los de mayor distancia de crowding
    devolver frente 1 de P

Funciones auxiliares a implementar aqui: dominates(a, b), fast_non_dominated_sort(F),
crowding_distance(F_front).
Validacion: comparar el frente con el de pymoo (NSGA2) usando la misma funcion objetivo.
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction  # noqa: F401
from .base import Result  # noqa: F401


def run(fitness: FitnessFunction, rng: np.random.Generator, pop_size: int = 40,
        crossover_rate: float = 0.9, mutation_rate: float | None = None) -> Result:
    # TODO (Persona A): implementar siguiendo el pseudocodigo de arriba.
    raise NotImplementedError("NSGA-II pendiente de implementar")
