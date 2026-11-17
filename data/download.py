"""Descarga (una vez) los datasets de OpenML a data/raw/.

Uso: python data/download.py [madelon] [arcene]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fsmeta.data import load_dataset  # noqa: E402

if __name__ == "__main__":
    for name in sys.argv[1:] or ["madelon"]:
        X, y, _ = load_dataset(name)
        print(f"{name}: {X.shape[0]} muestras, {X.shape[1]} variables, clases {sorted(set(y))}")
