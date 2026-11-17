"""Carga de datasets y particiones fijas train/test.

Datasets disponibles:
    - "breast_cancer": 569 muestras, 30 variables. Incluido en scikit-learn (no necesita
      descarga). Se usa para depurar: todo corre en segundos.
    - "madelon": dataset principal (NIPS 2003 Feature Selection Challenge). 500 variables,
      ~20 relevantes. Se descarga de OpenML la primera vez y se guarda en data/raw/.
    - "arcene": alta dimension (~10.000 variables, 200 muestras). Opcional.

La particion de test se hace UNA vez con una semilla fija (SPLIT_SEED) y nunca se usa
durante la busqueda: solo para la evaluacion final.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SPLIT_SEED = 42
TEST_SIZE = 0.30

# Nombre del dataset en OpenML (se unen las particiones train + valid del NIPS 2003,
# porque las etiquetas del test oficial no son publicas).
OPENML_NAMES = {
    "madelon": "madelon",
    "arcene": "arcene",
}


@dataclass
class Problem:
    """Un problema de seleccion de caracteristicas ya particionado."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Devuelve X, y (etiquetas 0..k-1) y nombres de variables."""
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data.astype(float), d.target.astype(int), list(d.feature_names)

    if name in OPENML_NAMES:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        d = fetch_openml(name=OPENML_NAMES[name], version=1, as_frame=False,
                         data_home=str(RAW_DIR), parser="auto")
        X = np.asarray(d.data, dtype=float)
        _, y = np.unique(np.asarray(d.target), return_inverse=True)
        names = list(d.feature_names) if d.feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
        return X, y.astype(int), names

    raise ValueError(f"Dataset desconocido: {name!r}. Opciones: breast_cancer, {', '.join(OPENML_NAMES)}")


def make_problem(name: str, test_size: float = TEST_SIZE, seed: int = SPLIT_SEED) -> Problem:
    """Carga el dataset y hace la particion estratificada train/test fija."""
    X, y, names = load_dataset(name)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return Problem(name, X_tr, y_tr, X_te, y_te, names)
