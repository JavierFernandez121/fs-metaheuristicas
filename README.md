# Selección de características con metaheurísticas (B6)

Proyecto de **Optimización y Heurística** (Grado en Ciencia e Ingeniería de Datos, ULPGC, 2026-2027).

Elegimos el mejor subconjunto de variables para un clasificador con metaheurísticas de
implementación propia (algoritmo genético, PSO binario, recocido simulado, búsqueda tabú y
NSGA-II biobjetivo) y las comparamos con métodos clásicos (selección hacia delante,
información mutua y LASSO) bajo el mismo presupuesto de evaluaciones.

¿Términos nuevos? Ver [`docs/glosario.md`](docs/glosario.md).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rápido

```bash
pytest -q                                              # pruebas (segundos)
python data/download.py madelon                        # descarga Madelon a data/raw/
python experiments/run_all.py --datasets breast_cancer --seeds 0 1 2 --budget 200   # prueba rápida
python experiments/run_all.py --jobs 4                 # experimentos completos (config.py), 4 a la vez
python experiments/make_tables.py                      # results/tables_<dataset>.md
python experiments/make_figures.py                     # figures/*.png
```

Toda la configuración (datasets, semillas, presupuesto, parámetros) está en
`experiments/config.py`.

## Estructura

```
├── data/                  download.py y README con origen y licencia de los datos
├── src/fsmeta/
│   ├── data.py            carga de datasets y partición train/test fija (70/30)
│   ├── fitness.py         función objetivo común: wrapper + CV + caché + presupuesto
│   ├── baselines.py       clásicos: todas, SFS, información mutua, LASSO
│   ├── stats.py           Friedman, Wilcoxon-Holm, frente de Pareto, hipervolumen
│   └── metaheuristics/
│       ├── base.py        interfaz común (Result) y utilidades
│       ├── random_search.py   línea base (implementada, sirve de ejemplo)
│       ├── ga.py  bpso.py  sa.py  tabu.py  nsga2.py   (pseudocódigo + TODO)
├── experiments/           config.py, run_all.py, make_tables.py, make_figures.py
├── tests/                 pruebas rápidas de interfaz y estadística
├── results/               CSV de cada ejecución + curvas de convergencia
├── figures/               figuras generadas por scripts
├── docs/                  glosario.md, registro_ia.md
└── report/                memoria.md (borrador por secciones)
```

## Cómo añadir un método

1. Implementa `run(fitness, rng, **params) -> Result` en `src/fsmeta/metaheuristics/<metodo>.py`
   (mira `random_search.py`).
2. Llama a `fitness(mask)` para evaluar (o `fitness.multi(mask)` en NSGA-II); captura
   `BudgetExhausted` para parar; usa solo `rng` para el azar.
3. Devuelve `Result.from_fitness(fitness)`.
4. `pytest -q` comprobará que cumple la interfaz y no se pasa del presupuesto.

## Reparto

| Persona A · Algoritmos | Persona B · Banco de pruebas |
| --- | --- |
| `ga.py`, `bpso.py`, `sa.py`, `tabu.py`, `nsga2.py` (+ sustituto) | `data.py`, `fitness.py`, `baselines.py`, `stats.py`, `experiments/` |

Flujo Git: una rama por persona (`persona-a`, `persona-b`), *pull request* o *merge* a `main`
cada día. Antes de hacer merge: `pytest -q`.

## Reproducibilidad

- Partición train/test fija (`SPLIT_SEED = 42`); el test no se usa durante la búsqueda.
- Cada ejecución usa `np.random.default_rng(seed)`; semillas en `config.py`.
- Tablas y figuras se regeneran con los scripts de `experiments/`.

## Uso de IA

Registro en [`docs/registro_ia.md`](docs/registro_ia.md); resumen en la sección 11 de la memoria.
