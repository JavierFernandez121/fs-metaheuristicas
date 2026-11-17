"""Analisis estadistico y multiobjetivo.   Responsable: Persona B

- friedman: hay diferencias entre todos los metodos?
- pairwise_wilcoxon: que pares difieren, con correccion de Holm
- pareto_front / hypervolume_2d: para comparar frentes (NSGA-II y nubes de puntos)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def friedman(df: pd.DataFrame, metric: str = "test_acc") -> tuple[float, float]:
    """df con columnas method, seed, <metric>. Bloques = semillas. Devuelve (estadistico, p)."""
    wide = df.pivot_table(index="seed", columns="method", values=metric).dropna()
    if wide.shape[1] < 3:
        raise ValueError("Friedman necesita al menos 3 metodos")
    return stats.friedmanchisquare(*[wide[c].values for c in wide.columns])


def pairwise_wilcoxon(df: pd.DataFrame, metric: str = "test_acc") -> pd.DataFrame:
    """Wilcoxon de rangos con signo por pares, emparejando por semilla, p-valores Holm."""
    wide = df.pivot_table(index="seed", columns="method", values=metric).dropna()
    rows = []
    for a, b in combinations(wide.columns, 2):
        diff = wide[a] - wide[b]
        p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(wide[a], wide[b]).pvalue
        rows.append({"A": a, "B": b, "median_diff": float(diff.median()), "p": float(p)})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(out)
    adj = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(out["p"])])
    out["p_holm"] = adj
    return out


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Filtra los puntos no dominados (minimizacion en todas las columnas)."""
    pts = np.asarray(points, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if keep[i]:
            dominated = np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)
            keep &= ~dominated
    return pts[keep]


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    """Hipervolumen (area dominada) de un frente 2D de minimizacion respecto a `ref`.

    Normalizar antes los objetivos (p. ej. n_variables / n) para que sea comparable."""
    front = pareto_front(points)
    front = front[(front[:, 0] <= ref[0]) & (front[:, 1] <= ref[1])]
    if len(front) == 0:
        return 0.0
    front = front[np.argsort(front[:, 0])]
    area, prev_y = 0.0, ref[1]
    for x, y in front:
        area += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(area)
