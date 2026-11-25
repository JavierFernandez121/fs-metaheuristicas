"""Recocido simulado (SA) para seleccion de caracteristicas.   Responsable: Persona A

Un solo candidato x que se mueve por vecinos (invertir entre 1 y k_max bits). Acepta siempre
las mejoras y los empeoramientos con probabilidad exp(-delta/T). La temperatura se multiplica
por `cooling` cada `iters_per_temp` iteraciones (enfriamiento geometrico por escalones).

Pseudocodigo
------------
    x <- mascara aleatoria; f_x <- fitness(x)
    si T0 es None:  T0 <- media de |delta f| de ~20 vecinos de x  (un empeoramiento medio se
                    acepta al principio con prob. ~ e^-1 = 37 %)
    T <- T0
    mientras quede presupuesto:
        repetir iters_per_temp veces:
            y <- vecino de x (invertir de 1 a k_max bits); f_y <- fitness(y)
            si f_y <= f_x o aleatorio < exp(-(f_y - f_x)/T):  x <- y
        T <- cooling * T
    devolver Result.from_fitness(fitness)

Parametros: T0=None (autocalibrada; las evaluaciones de calibracion gastan presupuesto y
cuentan para el mejor global), cooling=0.95, iters_per_temp=20, k_max=5, p_init=None
Ajuste (Breast Cancer, train, 5 semillas, 3000 evals): k_max=5 mejora f en 0.0013 frente a k_max=3
(error estandar 0.0001); cooling=0.99 empeora (enfria demasiado despacio para este presupuesto).
Con presupuesto 3000 hay ~150 escalones de temperatura: T final ~ T0 * 0.95**150 ~ T0/2000.
Complejidad por iteracion: O(coste_evaluacion).
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, flip, random_mask


def run(fitness: FitnessFunction, rng: np.random.Generator, T0: float | None = None,
        cooling: float = 0.95, iters_per_temp: int = 20, k_max: int = 5,
        p_init: float | None = None) -> Result:
    n = fitness.n_features
    guard = StallGuard(fitness)

    def neighbor(x):
        y = flip(x, rng, int(rng.integers(1, min(k_max, n) + 1)))
        if not y.any():
            y[rng.integers(n)] = True
        return y

    try:
        x = random_mask(n, rng, rng.uniform(0.02, 0.5) if p_init is None else p_init)
        fx = fitness(x)
        guard.tick()
        T = T0
        if T is None:  # calibracion: tamano tipico de un empeoramiento
            deltas = []
            for _ in range(20):
                fy = fitness(neighbor(x))
                guard.tick()
                deltas.append(abs(fy - fx))
            T = max(float(np.mean(deltas)), 1e-6)
        while True:
            for _ in range(iters_per_temp):
                y = neighbor(x)
                fy = fitness(y)
                guard.tick()
                d = fy - fx
                if d <= 0 or rng.random() < np.exp(-d / T):
                    x, fx = y, fy
            T *= cooling
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)
