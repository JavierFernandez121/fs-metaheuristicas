# Datos

| Dataset | Variables | Uso en el proyecto | Origen |
| --- | --- | --- | --- |
| Breast Cancer Wisconsin | 30 | Depuracion: todo corre en segundos | Incluido en scikit-learn (`load_breast_cancer`) |
| Madelon | 500 (unas 20 relevantes) | Dataset principal | NIPS 2003 Feature Selection Challenge, UCI; se descarga de OpenML |
| Arcene | ~10.000 | Opcional, alta dimension | NIPS 2003 Feature Selection Challenge, UCI; se descarga de OpenML |

Las etiquetas del test oficial del NIPS 2003 no son publicas, asi que se usan las particiones
etiquetadas (train + valid) y se hace una particion propia estratificada 70/30 con semilla
fija (`fsmeta/data.py`, `SPLIT_SEED = 42`). El 30 % de test solo se usa en la evaluacion final.

Descarga: `python data/download.py madelon arcene` (se guarda en `data/raw/`, que no se sube a Git).

TODO (Persona B): confirmar en las fichas de UCI y OpenML el numero de muestras, la licencia
y la cita correcta de cada dataset, y anotarlos aqui y en la seccion 8 de la memoria.
