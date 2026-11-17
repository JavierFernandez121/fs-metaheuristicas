"""Interfaz comun que deben cumplir TODOS los metodos de busqueda.

Cada metodo es una funcion con esta firma:

    def run(fitness: FitnessFunction, rng: np.random.Generator, **params) -> Result

- `fitness(mask)` devuelve la nota monoobjetivo (menor = mejor).
- `fitness.multi(mask)` devuelve (error, n_variables) para NSGA-II.
- `rng` es el generador aleatorio: usar SOLO este (nada de np.random.* global) para
  que la semilla haga el experimento reproducible.
- El metodo debe parar cuando se lance `BudgetExhausted` y devolver lo mejor que tenga.

`Result.from_fitness` construye el resultado a partir de lo que ha ido guardando la
funcion objetivo (mejor mascara y curva de convergencia), asi cada metodo no tiene que
llevar la cuenta por su cuenta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    best_mask: np.ndarray
    best_f: float
    n_evals: int
    history: list[tuple[int, float]] = field(default_factory=list)
    # Solo para metodos multiobjetivo: lista de (error, n_variables, mascara)
    pareto: list[tuple[float, int, np.ndarray]] | None = None

    @classmethod
    def from_fitness(cls, fitness, pareto=None) -> "Result":
        mask = fitness.best_mask
        if mask is None:
            mask = np.ones(fitness.n_features, dtype=bool)
        return cls(mask, float(fitness.best_f), fitness.n_evals, list(fitness.history), pareto)


def random_mask(n: int, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Mascara aleatoria con probabilidad p de incluir cada variable (nunca vacia)."""
    m = rng.random(n) < p
    if not m.any():
        m[rng.integers(n)] = True
    return m


def flip(mask: np.ndarray, rng: np.random.Generator, k: int = 1) -> np.ndarray:
    """Vecino: copia de la mascara con k bits invertidos al azar."""
    new = mask.copy()
    idx = rng.choice(mask.size, size=k, replace=False)
    new[idx] = ~new[idx]
    return new
