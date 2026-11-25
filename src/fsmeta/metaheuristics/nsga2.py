"""NSGA-II (Deb et al., 2002) para el problema biobjetivo.   Responsable: Persona A

Objetivos (ambos a minimizar): fitness.multi(mask) -> (error CV, numero de variables).
Devuelve el frente de Pareto aproximado en Result.pareto como lista de
(error, n_variables, mascara), ordenada por numero de variables.

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

Funciones auxiliares: dominates(a, b), fast_non_dominated_sort(F), crowding_distance(F_front).
Validacion: comparar el frente con el de pymoo (NSGA2) usando la misma funcion objetivo.

Nota de implementacion: `fitness.multi` no actualiza el mejor global ni la curva de
convergencia; por eso cada individuo se evalua tambien con `fitness(mask)` (sale de la cache,
no gasta presupuesto) para que Result.best_f e history sean comparables con los demas metodos.
Complejidad por generacion: O(M N^2) en la ordenacion (M=2 objetivos, N=2*pop_size) + evaluaciones.
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, random_mask


def dominates(a, b) -> bool:
    """a domina a b (minimizacion): no es peor en ningun objetivo y es mejor en alguno."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F):
    """F: (N, m). Devuelve (rank, fronts); rank 0 = frente de Pareto.

    Version vectorizada: construye de golpe la matriz dom[i, j] = "i domina a j" y va
    retirando frentes (equivale al algoritmo de Deb et al., pero sin bucles en Python)."""
    F = np.asarray(F, dtype=float)
    N = len(F)
    dom = (F[:, None, :] <= F[None, :, :]).all(axis=2) & (F[:, None, :] < F[None, :, :]).any(axis=2)
    count = dom.sum(axis=0)                     # cuantas soluciones dominan a j
    rank = np.zeros(N, dtype=int)
    assigned = np.zeros(N, dtype=bool)
    fronts = []
    current = np.flatnonzero(count == 0)
    while current.size:
        rank[current] = len(fronts)
        assigned[current] = True
        fronts.append(current.tolist())
        count = count - dom[current].sum(axis=0)  # quitar lo que dominaba este frente
        current = np.flatnonzero((count == 0) & ~assigned)
    return rank, fronts


def crowding_distance(F):
    """Distancia de crowding dentro de un frente; los extremos reciben infinito."""
    m = len(F)
    d = np.zeros(m)
    if m <= 2:
        d[:] = np.inf
        return d
    for k in range(F.shape[1]):
        idx = np.argsort(F[:, k])
        span = F[idx[-1], k] - F[idx[0], k]
        d[idx[0]] = d[idx[-1]] = np.inf
        if span > 0:
            d[idx[1:-1]] += (F[idx[2:], k] - F[idx[:-2], k]) / span
    return d


def _rank_and_crowd(F):
    rank, fronts = fast_non_dominated_sort(F)
    crowd = np.zeros(len(F))
    for fr in fronts:
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd


def _survival(pop, F, size):
    """Frentes completos mientras quepan; el ultimo se corta por mayor crowding."""
    _, fronts = fast_non_dominated_sort(F)
    chosen = []
    for fr in fronts:
        if len(chosen) + len(fr) <= size:
            chosen += fr
        else:
            d = crowding_distance(F[fr])
            chosen += [fr[i] for i in np.argsort(-d)[: size - len(chosen)]]
            break
    return [pop[i] for i in chosen], F[chosen]


def run(fitness: FitnessFunction, rng: np.random.Generator, pop_size: int = 40,
        crossover_rate: float = 0.9, mutation_rate: float | None = None) -> Result:
    n = fitness.n_features
    p_mut = 1.0 / n if mutation_rate is None else mutation_rate
    guard = StallGuard(fitness)
    pop, F = [], np.empty((0, 2))

    def evaluate(mask):
        fitness(mask)            # actualiza mejor global e historial (cache: gratis)
        guard.tick()
        return fitness.multi(mask)

    def tournament(rank, crowd):  # torneo binario: menor rango; si empatan, mayor crowding
        i, j = rng.integers(len(rank), size=2)
        if rank[i] != rank[j]:
            return int(i if rank[i] < rank[j] else j)
        return int(i if crowd[i] >= crowd[j] else j)

    try:
        pop = [random_mask(n, rng, rng.uniform(0.05, 0.5)) for _ in range(pop_size)]
        F = np.array([evaluate(m) for m in pop], dtype=float)
        rank, crowd = _rank_and_crowd(F)
        while True:
            children = []
            while len(children) < pop_size:
                p1, p2 = pop[tournament(rank, crowd)], pop[tournament(rank, crowd)]
                if rng.random() < crossover_rate:
                    child = np.where(rng.random(n) < 0.5, p1, p2)
                else:
                    child = p1.copy()
                child ^= rng.random(n) < p_mut
                if not child.any():
                    child[rng.integers(n)] = True
                children.append(child)
            Fc = np.array([evaluate(c) for c in children], dtype=float)
            pop, F = _survival(pop + children, np.vstack([F, Fc]), pop_size)
            rank, crowd = _rank_and_crowd(F)
    except BudgetExhausted:
        pass
    pareto = None
    if len(pop) == len(F) and len(pop) > 0:
        rank, _ = fast_non_dominated_sort(F)
        seen, pareto = set(), []
        for i in np.flatnonzero(rank == 0):
            key = (float(F[i, 0]), int(F[i, 1]))
            if key not in seen:  # una sola mascara por punto del frente
                seen.add(key)
                pareto.append((key[0], key[1], pop[i].copy()))
        pareto.sort(key=lambda t: t[1])
    return Result.from_fitness(fitness, pareto=pareto)
