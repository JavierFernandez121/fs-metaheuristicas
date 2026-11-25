"""Genera las tablas de resultados y los tests estadisticos a partir de results/*.csv.

Uso: python experiments/make_tables.py
Salida: results/tables_<dataset>.md (para copiar a la memoria)

Contenido de cada tabla:
    1. Resumen por metodo: media ± desviacion tipica de precision en test, error CV, numero de
       variables, evaluaciones y tiempo, y la brecha de sobreajuste (error test - error CV).
    2. Tests estadisticos (metodos estocasticos, emparejados por semilla) sobre dos metricas:
       - cv_error: calidad de la busqueda (lo que optimizan los metodos)
       - test_acc: generalizacion (lo que importa)
       Friedman entre todos y Wilcoxon por pares con correccion de Holm.
    3. NSGA-II: hipervolumen del frente por semilla (objetivos error CV y |x|/n, referencia (1, 1)).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsmeta.stats import friedman, hypervolume_2d, pairwise_wilcoxon  # noqa: E402


def _ms(s: pd.Series, fmt: str) -> str:
    return f"{s.mean():{fmt}} ± {s.std() if len(s) > 1 else 0:{fmt}}"


def summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(gap=(1 - df.test_acc) - df.cv_error)
    rows = []
    for m, g in df.groupby("method"):
        rows.append({
            "method": m, "runs": len(g),
            "test_acc": _ms(g.test_acc, ".4f"), "cv_error": _ms(g.cv_error, ".4f"),
            "gap (err_test - err_cv)": _ms(g.gap, ".4f"),
            "n_selected": _ms(g.n_selected, ".1f"),
            "n_evals": int(round(g.n_evals.mean())), "time_s": round(g.time_s.mean(), 1),
            "_order": g.test_acc.mean(),
        })
    return pd.DataFrame(rows).sort_values("_order", ascending=False).drop(columns="_order").set_index("method")


def stats_block(df: pd.DataFrame, metric: str, label: str) -> list[str]:
    stochastic = df[df.groupby("method").seed.transform("nunique") > 1]
    if stochastic.method.nunique() < 3:
        return []
    stat, p = friedman(stochastic, metric)
    pw = pairwise_wilcoxon(stochastic, metric)
    return [f"## Tests sobre {metric} ({label})\n",
            f"Friedman (metodos estocasticos, {stochastic.seed.nunique()} semillas): "
            f"chi2 = {stat:.3f}, p = {p:.4g}\n",
            "Wilcoxon por pares (median_diff = mediana de A - B; p_holm corregido):\n",
            pw.to_markdown(index=False, floatfmt=".4g"), ""]


def hypervolume_block(dataset: str, df: pd.DataFrame) -> list[str]:
    rows = []
    for seed in sorted(df[df.method == "nsga2"].seed):
        p = ROOT / "results" / "histories" / f"{dataset}_nsga2_{seed}.json"
        if not p.exists():
            continue
        front = json.loads(p.read_text()).get("pareto") or []
        if not front:
            continue
        n = int(df.n_features.iloc[0])
        pts = np.array([[f["cv_error"], f["n_selected"] / n] for f in front])
        rows.append({"seed": seed, "puntos del frente": len(front),
                     "hipervolumen": hypervolume_2d(pts, ref=(1.0, 1.0)),
                     "min error CV": pts[:, 0].min(),
                     "max precision test": max(f["test_acc"] for f in front)})
    if not rows:
        return []
    t = pd.DataFrame(rows)
    return ["## NSGA-II: frente de Pareto por semilla\n",
            t.to_markdown(index=False, floatfmt=("g", "g", ".4f", ".4f", ".4f")),
            f"\nHipervolumen medio: {t.hipervolumen.mean():.4f} ± {t.hipervolumen.std():.4f}\n"]


def madelon_relevant(threshold: float = 0.3) -> set[int]:
    """Las 20 variables relevantes de Madelon (5 informativas + 15 combinaciones lineales) estan
    correladas entre si; las 480 de ruido son independientes. Se identifican en el entrenamiento
    como las variables cuya maxima |correlacion| con otra supera `threshold`."""
    from fsmeta.data import make_problem
    X = make_problem("madelon").X_train
    C = np.abs(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(C, 0)
    return set(np.flatnonzero(C.max(axis=1) > threshold).tolist())


def relevant_block(df: pd.DataFrame) -> list[str]:
    rel = madelon_relevant()
    sel = df.selected.fillna("").astype(str).map(lambda s: {int(v) for v in s.split()})
    d = df.assign(relevantes=sel.map(lambda s: len(s & rel)),
                  pureza=sel.map(lambda s: len(s & rel) / max(len(s), 1)))
    rows = []
    for m, g in d.groupby("method"):
        rows.append({"method": m, f"relevantes elegidas (de {len(rel)})": _ms(g.relevantes, ".1f"),
                     "variables elegidas": _ms(g.n_selected, ".1f"),
                     "% de las elegidas que son relevantes": _ms(100 * g.pureza, ".0f"),
                     "_o": g.pureza.mean()})
    t = pd.DataFrame(rows).sort_values("_o", ascending=False).drop(columns="_o").set_index("method")
    return ["## Madelon: variables relevantes recuperadas\n",
            f"Relevantes identificadas por correlacion en el entrenamiento: {sorted(rel)}\n",
            t.to_markdown(), ""]


def main():
    for csv in sorted((ROOT / "results").glob("*.csv")):
        df = pd.read_csv(csv)
        lines = [f"# Resultados: {csv.stem}\n", summary(df).to_markdown(), ""]
        if csv.stem == "madelon":
            lines += relevant_block(df)
        lines += stats_block(df, "cv_error", "calidad de la busqueda")
        lines += stats_block(df, "test_acc", "generalizacion")
        lines += hypervolume_block(csv.stem, df)
        out = ROOT / "results" / f"tables_{csv.stem}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
