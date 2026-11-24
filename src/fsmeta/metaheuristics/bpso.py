"""PSO binario (Kennedy y Eberhart, 1997) para seleccion de caracteristicas.   Responsable: Persona A

Cada particula tiene una mascara x y una velocidad real v (una por variable). La velocidad
se actualiza como en el PSO continuo y la posicion se re-muestrea con una funcion de
transferencia sigmoide: P(x_j = 1) = 1 / (1 + exp(-v_j)).

Pseudocodigo
------------
    inicializar n_particles mascaras y velocidades; evaluar; pbest <- x; gbest <- mejor pbest
    mientras quede presupuesto:
        para cada particula i:
            v <- w*v + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i);  v <- clip(v, -v_max, v_max)
            x_i[j] <- 1 con prob. sigmoide(v[j])
            evaluar x_i; actualizar pbest_i y gbest
    devolver Result.from_fitness(fitness)

Parametros: n_particles=30, w=0.7, c1=c2=1.5, v_max=4.0 (limita la velocidad para que la
probabilidad nunca sea exactamente 0 o 1: sigmoide(4) ~ 0.982), p_init=None (densidad
aleatoria por particula en [0.02, 0.5], da enjambres iniciales con tamanos variados)
Complejidad por iteracion: O(n_particles * (n + coste_evaluacion)).
"""
from __future__ import annotations

import numpy as np

from ..fitness import BudgetExhausted, FitnessFunction
from ._guard import StallGuard
from .base import Result, random_mask


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def run(fitness: FitnessFunction, rng: np.random.Generator, n_particles: int = 30,
        w: float = 0.7, c1: float = 1.5, c2: float = 1.5, v_max: float = 4.0,
        p_init: float | None = None) -> Result:
    n = fitness.n_features
    guard = StallGuard(fitness)
    try:
        x = np.array([random_mask(n, rng, rng.uniform(0.02, 0.5) if p_init is None else p_init)
                      for _ in range(n_particles)])
        v = rng.uniform(-1, 1, size=(n_particles, n))
        pbest_f = np.empty(n_particles)
        for i in range(n_particles):
            pbest_f[i] = fitness(x[i])
            guard.tick()
        pbest = x.copy()
        g = int(np.argmin(pbest_f))
        gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        while True:
            for i in range(n_particles):
                r1, r2 = rng.random(n), rng.random(n)
                xi = x[i].astype(float)
                v[i] = np.clip(w * v[i] + c1 * r1 * (pbest[i] - xi) + c2 * r2 * (gbest - xi),
                               -v_max, v_max)
                x[i] = rng.random(n) < _sigmoid(v[i])
                if not x[i].any():
                    x[i, rng.integers(n)] = True
                fi = fitness(x[i])
                guard.tick()
                if fi < pbest_f[i]:
                    pbest[i], pbest_f[i] = x[i].copy(), fi
                    if fi < gbest_f:
                        gbest, gbest_f = x[i].copy(), fi
    except BudgetExhausted:
        pass
    return Result.from_fitness(fitness)
