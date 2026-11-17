"""Genera las figuras de la memoria a partir de results/.

Uso: python experiments/make_figures.py
Salidas en figures/:
    convergence_<dataset>.png   curvas de convergencia (mediana y rango intercuartilico)
    boxplot_<dataset>.png       precision en test por metodo
    tradeoff_<dataset>.png      precision en test frente a numero de variables
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
HIST = ROOT / "results" / "histories"


def step_curve(history, budget):
    """Convierte [(n_evals, mejor_f), ...] en una curva escalonada sobre 1..budget."""
    grid = np.arange(1, budget + 1)
    if not history:
        return grid, np.full(budget, np.nan)
    ev, f = np.array(history).T
    idx = np.searchsorted(ev, grid, side="right") - 1
    y = np.where(idx >= 0, f[np.clip(idx, 0, None)], np.nan)
    return grid, y


def convergence(df, ds):
    budget = int(df.n_evals.max())
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in sorted(df.method.unique()):
        curves = []
        for seed in df[df.method == method].seed:
            p = HIST / f"{ds}_{method}_{seed}.json"
            if p.exists():
                curves.append(step_curve(json.loads(p.read_text())["history"], budget)[1])
        if not curves or len(df[df.method == method]) < 2:
            continue
        c = np.array(curves)
        x = np.arange(1, budget + 1)
        med = np.nanmedian(c, axis=0)
        ax.plot(x, med, label=method)
        ax.fill_between(x, np.nanpercentile(c, 25, axis=0), np.nanpercentile(c, 75, axis=0), alpha=0.2)
    ax.set(xlabel="Evaluaciones de la funcion objetivo", ylabel="Mejor f(x) (menor es mejor)",
           title=f"Convergencia - {ds}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"convergence_{ds}.png", dpi=150)
    plt.close(fig)


def boxplot(df, ds):
    order = df.groupby("method").test_acc.median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df[df.method == m].test_acc for m in order], tick_labels=order)
    ax.set(ylabel="Precision en test", title=f"Precision en test por metodo - {ds}")
    fig.tight_layout()
    fig.savefig(FIG / f"boxplot_{ds}.png", dpi=150)
    plt.close(fig)


def tradeoff(df, ds):
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in sorted(df.method.unique()):
        d = df[df.method == method]
        ax.scatter(d.n_selected, d.test_acc, label=method, alpha=0.7)
    ax.set(xlabel="Numero de variables seleccionadas", ylabel="Precision en test",
           title=f"Precision frente a numero de variables - {ds}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"tradeoff_{ds}.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for csv in sorted((ROOT / "results").glob("*.csv")):
        df = pd.read_csv(csv)
        ds = csv.stem
        convergence(df, ds)
        boxplot(df, ds)
        tradeoff(df, ds)
        print(f"-> figuras de {ds} en figures/")


if __name__ == "__main__":
    main()
