# Seleccion de caracteristicas con metaheuristicas

**Autores:** Persona A, Persona B
**Asignatura:** Optimizacion y Heuristica · Grado en Ciencia e Ingenieria de Datos · ULPGC · 2026-2027

> Borrador de la memoria. Cada seccion indica su responsable. Las cifras salen SIEMPRE de
> `results/` (generadas por `experiments/`), nunca escritas a mano.

## 1. Resumen  ·  Ambos (al final)
(< 300 palabras)

## 2. Introduccion y motivacion  ·  Ambos

## 3. Descripcion del problema y estado del arte  ·  A: metaheuristicas y NSGA-II · B: filtros, wrappers y embebidos
Referencias reales y consultadas (guardar DOI o enlace).

## 4. Formulacion matematica  ·  A
- Variable de decision: x en {0,1}^n
- Monoobjetivo: f(x) = alpha * err_CV(x) + (1 - alpha) * |x| / n, con |x| >= 1
- Biobjetivo: min (err_CV(x), |x|)
- Tipo de problema: discreto, combinatorio (2^n soluciones), no convexo, funcion objetivo
  ruidosa y cara de evaluar (cada evaluacion entrena un modelo).

## 5. Metodos  ·  A (metaheuristicas) · B (clasicos)
Para cada metodo: idea, pseudocodigo, parametros, complejidad y por que se eligio.

## 6. Implementacion  ·  B
Organizacion del codigo (interfaz comun, funcion objetivo con cache y presupuesto),
decisiones de diseno y dificultades.

## 7. Datos  ·  B
Origen, licencia, descripcion, preprocesado, particion 70/30 fija.

## 8. Experimentacion y resultados  ·  B
Diseno (presupuesto, semillas, clasificador), metricas, tablas (`results/tables_*.md`),
figuras (`figures/`), tests estadisticos, frente de Pareto e hipervolumen.

## 9. Discusion  ·  Ambos
Que metodo funciona mejor, cuando y por que. Sobreajuste (error CV frente a test). Limitaciones.

## 10. Conclusiones y trabajo futuro  ·  Ambos

## 11. Uso de la IA  ·  Ambos
A partir de `docs/registro_ia.md`.

## 12. Referencias y anexos
