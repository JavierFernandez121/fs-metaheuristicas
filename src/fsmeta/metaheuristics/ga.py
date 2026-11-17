"""Algoritmo genetico (GA) para seleccion de caracteristicas.   Responsable: Persona A

Idea: una poblacion de mascaras evoluciona por seleccion, cruce y mutacion.

Pseudocodigo
------------
    P <- pop_size mascaras aleatorias; evaluar cada una con fitness(.)
    mientras quede presupuesto:
        elite <- los `elitism` mejores de P
        hijos <- []
        mientras len(hijos) < pop_size - elitism:
            p1, p2 <- torneo(P, k=tournament_k), torneo(P, k=tournament_k)
            h <- cruce_uniforme(p1, p2) con prob. crossover_rate (si no, copia de p1)
            h <- mutacion bit-flip: cada bit se invierte con prob. mutation_rate
            evaluar h; anadir a hijos
        P <- elite + hijos
    devolver Result.from_fitness(fitness)

Parametros a justificar en la memoria
-------------------------------------
    pop_size=30, crossover_rate=0.9, mutation_rate=1/n, tournament_k=3, elitism=2,
    p_init=0.5 (prob. de incluir cada variable al inicializar)

Complejidad por generacion: O(pop_size * coste_evaluacion).
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction  # noqa: F401
from .base import Result, random_mask  # noqa: F401


def run(fitness: FitnessFunction, rng: np.random.Generator, pop_size: int = 30,
        crossover_rate: float = 0.9, mutation_rate: float | None = None,
        tournament_k: int = 3, elitism: int = 2, p_init: float = 0.5) -> Result:
    # TODO (Persona A): implementar siguiendo el pseudocodigo de arriba.
    # Recuerda: envolver el bucle en `try: ... except BudgetExhausted: pass`.
    raise NotImplementedError("GA pendiente de implementar")
