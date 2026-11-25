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
    p_init=0.5 (prob. de incluir cada variable al inicializar; None = densidad aleatoria
    por individuo en [0.02, 0.5], da poblaciones iniciales con tamanos variados)

Complejidad por generacion: O(pop_size * coste_evaluacion).
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, random_mask


def run(fitness: FitnessFunction, rng: np.random.Generator, pop_size: int = 30,
        crossover_rate: float = 0.9, mutation_rate: float | None = None,
        tournament_k: int = 3, elitism: int = 2, p_init: float | None = 0.5) -> Result:
    n = fitness.n_features
    p_mut = 1.0 / n if mutation_rate is None else mutation_rate
    guard = StallGuard(fitness)

    def evaluate(mask):
        f = fitness(mask)
        guard.tick()
        return f

    def tournament(fit):
        idx = rng.integers(len(fit), size=tournament_k)
        return int(idx[np.argmin(fit[idx])])

    try:
        pop = [random_mask(n, rng, rng.uniform(0.02, 0.5) if p_init is None else p_init)
               for _ in range(pop_size)]
        fit = np.array([evaluate(m) for m in pop])
        while True:
            order = np.argsort(fit)
            new_pop = [pop[i].copy() for i in order[:elitism]]
            new_fit = [fit[i] for i in order[:elitism]]  # la elite no se reevalua
            while len(new_pop) < pop_size:
                p1, p2 = pop[tournament(fit)], pop[tournament(fit)]
                if rng.random() < crossover_rate:  # cruce uniforme
                    child = np.where(rng.random(n) < 0.5, p1, p2)
                else:
                    child = p1.copy()
                child ^= rng.random(n) < p_mut     # mutacion bit-flip
                if not child.any():
                    child[rng.integers(n)] = True  # restriccion |x| >= 1
                new_pop.append(child)
                new_fit.append(evaluate(child))
            pop, fit = new_pop, np.array(new_fit)
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)
