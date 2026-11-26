# Registro de uso de IA

La memoria debe incluir un apartado obligatorio sobre el uso de IA: herramientas, para que
tareas, ejemplos de las peticiones mas relevantes, valoracion critica y errores detectados.
Apuntad aqui cada uso relevante sobre la marcha y el apartado se escribira solo.

| Fecha | Quien | Herramienta | Para que | Peticion (resumen) | Resultado / errores detectados |
| --- | --- | --- | --- | --- | --- |
| 2026-11-17 | Ambos | Claude | Planificacion | Resumen del enunciado B6, plan de trabajo, reparto y calendario | Plan en el documento compartido; revisado por ambos |
| 2026-11-17 | Ambos | Claude | Comprension | Glosario de conceptos explicado desde cero | docs/glosario.md |
| 2026-11-17 | Ambos | Claude | Codigo | Estructura del repositorio, funcion objetivo, linea base, clasicos, scripts de experimentos y esqueletos con pseudocodigo de las metaheuristicas | Revisar y entender todo el codigo antes de usarlo |
| 2026-11-25 | B | Claude | Revision de codigo | Revisar el codigo de Persona A y el fallo de bucle de random_search | Bucle reproducido (6 variables) y corregido con StallGuard + prueba; detectados 2 fallos logicos en tabu.py (aspiracion y deteccion de mejora), comprobados con una copia instrumentada |
| 2026-11-25 | B | Claude | Experimentos | Ejecucion en paralelo (--jobs) y carga de Madelon desde el zip de UCI | Estimacion de coste: ~50 ms por evaluacion en un problema del tamano de Madelon |
| 2026-11-25 | B | Claude | Memoria | Borrador de 5.11 (clasicos), 6 (implementacion) y 7 (datos) con referencias reales | Referencias comprobadas en sus paginas (DOI, revista, paginas); falta leerlas y revisar el texto |
| 2026-11-25 | B | Claude | Experimentos y analisis | Ejecutar Breast Cancer y Madelon (10 semillas, 3000 evaluaciones), tablas, tests y figuras | Ampliados make_tables.py (tests sobre error CV y test, hipervolumen, variables relevantes de Madelon) y make_figures.py (Pareto); cifras de la seccion 8 comprobadas contra las tablas (se corrigieron 3 afirmaciones que no cuadraban con los datos) |
| 2026-11-25 | B | Claude | Memoria | Borrador de 3 (mitad B), 8 y referencias de B | Pendiente: leer las referencias y revisar el texto |
