"""Genera las figuras de la memoria a partir de results/.

Uso: python experiments/make_figures.py
Salidas en figures/:
    convergence_<dataset>.png   curvas de convergencia (mediana y rango intercuartilico)
    boxplot_<dataset>.png       error CV (busqueda) frente a precision en test (generalizacion)
    tradeoff_<dataset>.png      precision en test frente a numero de variables
    pareto_<dataset>.png        frentes de Pareto de NSGA-II (error CV y error de test)

Colores fijos por metodo (no cambian entre figuras): las 5 metaheuristicas y la busqueda
aleatoria tienen color propio; los metodos clasicos van en gris con marcadores distintos.
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

COLORS = {"ga": "#257de0", "bpso": "#f47230", "sa": "#00b894", "tabu": "#f2a600",
          "nsga2": "#6a4ce2", "random": "#ec4647"}
GRAY = "#7a7a7a"
MARKERS = {"all": "s", "sfs": "^", "mi": "D", "lasso": "v"}
NAMES = {"ga": "GA", "bpso": "BPSO", "sa": "SA", "tabu": "Tabu", "nsga2": "NSGA-II",
         "random": "Aleatoria", "all": "Todas", "sfs": "SFS", "mi": "Info. mutua", "lasso": "LASSO"}
ORDER = ["ga", "bpso", "sa", "tabu", "nsga2", "random", "sfs", "mi", "lasso", "all"]

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.edgecolor": "#9c9c9c", "axes.axisbelow": True, "legend.frameon": False,
})


def color(m):
    return COLORS.get(m, GRAY)


def methods_in(df):
    return [m for m in ORDER if m in set(df.method)]


def step_curve(history, budget):
    """Convierte [(n_evals, mejor_f), ...] en una curva escalonada sobre 1..budget."""
    grid = np.arange(1, budget + 1)
    if not history:
        return np.full(budget, np.nan)
    ev, f = np.array(history).T
    idx = np.searchsorted(ev, grid, side="right") - 1
    return np.where(idx >= 0, f[np.clip(idx, 0, None)], np.nan)


def convergence(df, ds):
    budget = int(df.n_evals.max())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(1, budget + 1)
    for m in [m for m in methods_in(df) if m in COLORS]:  # solo busquedas iterativas
        seeds = df[df.method == m].seed
        curves = [step_curve(json.loads((HIST / f"{ds}_{m}_{s}.json").read_text())["history"], budget)
                  for s in seeds if (HIST / f"{ds}_{m}_{s}.json").exists()]
        c = np.array(curves)
        med = np.nanmedian(c, axis=0)
        ax.plot(x, med, color=color(m), lw=2, label=NAMES[m])
        ax.fill_between(x, np.nanpercentile(c, 25, axis=0), np.nanpercentile(c, 75, axis=0),
                        color=color(m), alpha=0.15, lw=0)
    ax.set(xlabel="Evaluaciones de la funcion objetivo", ylabel="Mejor f(x) encontrada (menor es mejor)",
           title=f"Convergencia - {ds} (mediana y rango intercuartilico)")
    lines = ax.get_lines()
    if lines:  # recorta el arranque (primeras evaluaciones) para que se vea el final
        y_low = min(np.nanmin(l.get_ydata()) for l in lines)
        y_high = max(l.get_ydata()[budget // 20] for l in lines)
        ax.set_ylim(y_low - 0.05 * (y_high - y_low), y_high)
    ax.legend(ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / f"convergence_{ds}.png", dpi=160)
    plt.close(fig)


def boxplot(df, ds):
    ms = methods_in(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, (col, label) in zip(axes, [("cv_error", "Error de validacion cruzada (busqueda)"),
                                       ("test_acc", "Precision en test (generalizacion)")]):
        data = [df[df.method == m][col].values for m in ms]
        bp = ax.boxplot(data, tick_labels=[NAMES[m] for m in ms], patch_artist=True, widths=0.6,
                        medianprops={"color": "#111111", "lw": 1.5})
        for patch, m in zip(bp["boxes"], ms):
            patch.set(facecolor=color(m), alpha=0.35, edgecolor=color(m), lw=1.2)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("menor es mejor")
    axes[1].set_ylabel("mayor es mejor")
    fig.suptitle(f"{ds}: lo que optimiza la busqueda frente a lo que generaliza", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / f"boxplot_{ds}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def tradeoff(df, ds):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in methods_in(df):
        d = df[df.method == m]
        ax.scatter(d.n_selected, d.test_acc, s=36, color=color(m), marker=MARKERS.get(m, "o"),
                   alpha=0.8, edgecolor="white", lw=0.8, label=NAMES[m])
    if df.n_features.iloc[0] > 100:
        ax.set_xscale("log")
    ax.set(xlabel="Numero de variables seleccionadas", ylabel="Precision en test",
           title=f"Precision frente a numero de variables - {ds} (un punto por ejecucion)")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / f"tradeoff_{ds}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def pareto(df, ds):
    seeds = sorted(df[df.method == "nsga2"].seed)
    fronts = []
    for s in seeds:
        p = HIST / f"{ds}_nsga2_{s}.json"
        if p.exists() and json.loads(p.read_text()).get("pareto"):
            fronts.append(pd.DataFrame(json.loads(p.read_text())["pareto"]))
    if not fronts:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, f in enumerate(fronts):
        f = f.sort_values("n_selected")
        ax.step(f.n_selected, f.cv_error, where="post", color=COLORS["nsga2"], alpha=0.35, lw=1.2,
                label="Frente NSGA-II (error CV), una linea por semilla" if i == 0 else None)
        ax.scatter(f.n_selected, 1 - f.test_acc, color=COLORS["nsga2"], marker="x", s=22, alpha=0.6,
                   label="Mismas soluciones, error en test" if i == 0 else None)
    for m in ["ga", "bpso", "sa", "tabu"]:
        d = df[df.method == m]
        if len(d):
            ax.scatter(d.n_selected.mean(), d.cv_error.mean(), color=color(m), s=60, marker="o",
                       edgecolor="white", lw=1, zorder=3, label=f"{NAMES[m]} (media, error CV)")
    if df.n_features.iloc[0] > 100:
        ax.set_xscale("log")
    ax.set(xlabel="Numero de variables |x|", ylabel="Error (menor es mejor)",
           title=f"Compromiso error-tamano - {ds}")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / f"pareto_{ds}.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for csv in sorted((ROOT / "results").glob("*.csv")):
        df = pd.read_csv(csv)
        ds = csv.stem
        convergence(df, ds)
        boxplot(df, ds)
        tradeoff(df, ds)
        pareto(df, ds)
        print(f"-> figuras de {ds} en figures/")


if __name__ == "__main__":
    main()
