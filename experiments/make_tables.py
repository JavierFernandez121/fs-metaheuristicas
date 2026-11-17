"""Genera las tablas de resultados y los tests estadisticos a partir de results/*.csv.

Uso: python experiments/make_tables.py
Salida: results/tables_<dataset>.md (para copiar a la memoria)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsmeta.stats import friedman, pairwise_wilcoxon  # noqa: E402


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("method")
    out = pd.DataFrame({
        "runs": g.size(),
        "test_acc": g.test_acc.mean().map("{:.4f}".format) + " ± " + g.test_acc.std().fillna(0).map("{:.4f}".format),
        "cv_error": g.cv_error.mean().map("{:.4f}".format) + " ± " + g.cv_error.std().fillna(0).map("{:.4f}".format),
        "n_selected": g.n_selected.mean().map("{:.1f}".format) + " ± " + g.n_selected.std().fillna(0).map("{:.1f}".format),
        "n_evals": g.n_evals.mean().round(0).astype(int),
        "time_s": g.time_s.mean().round(1),
    })
    return out.sort_values("test_acc", ascending=False)


def main():
    for csv in sorted((ROOT / "results").glob("*.csv")):
        df = pd.read_csv(csv)
        lines = [f"# Resultados: {csv.stem}\n", summary(df).to_markdown(), ""]
        stochastic = df[df.groupby("method").seed.transform("nunique") > 1]
        if stochastic.method.nunique() >= 3:
            stat, p = friedman(stochastic)
            lines += [f"Friedman (metodos estocasticos, test_acc): chi2 = {stat:.3f}, p = {p:.4g}\n",
                      pairwise_wilcoxon(stochastic).to_markdown(index=False), ""]
        out = ROOT / "results" / f"tables_{csv.stem}.md"
        out.write_text("\n".join(lines))
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
