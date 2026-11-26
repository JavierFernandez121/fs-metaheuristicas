# Seleccion de caracteristicas con metaheuristicas

**Autores:** Persona A, Persona B
**Asignatura:** Optimizacion y Heuristica · Grado en Ciencia e Ingenieria de Datos · ULPGC · 2026-2027

> Borrador de la memoria. Cada seccion indica su responsable. Las cifras salen SIEMPRE de
> `results/` (generadas por `experiments/`), nunca escritas a mano.

## 1. Resumen  ·  Ambos (al final)
(< 300 palabras)

## 2. Introduccion y motivacion  ·  Ambos

## 3. Descripcion del problema y estado del arte  ·  A: metaheuristicas y NSGA-II · B: filtros, wrappers y embebidos

### 3.A Estado del arte: metaheuristicas y NSGA-II  ·  A

#### 3.A.1 Seleccion de caracteristicas como busqueda de subconjuntos

La seleccion de caracteristicas busca reducir la dimension de los datos manteniendo o mejorando el
rendimiento de un clasificador, y es dificil sobre todo por el tamano del espacio de busqueda: hay `2^n`
subconjuntos para `n` variables (Xue et al., 2016). Los metodos se agrupan en filtros, que valoran los
subconjuntos con criterios independientes del clasificador, y wrappers, que incluyen el clasificador en
la evaluacion y suelen conseguir mejor rendimiento para ese clasificador concreto, a costa de un mayor
coste computacional (Xue et al., 2016). Kohavi y John (1997) formalizaron el enfoque wrapper: la
busqueda se adapta al algoritmo de aprendizaje y al dominio, y lo compararon con un filtro (Relief), con
mejoras de precision en algunos conjuntos de datos. Este trabajo adopta el enfoque wrapper (seccion 4.2);
como cada subconjunto candidato exige entrenar un modelo, todos los metodos se comparan con el mismo
presupuesto de evaluaciones.

#### 3.A.2 Metaheuristicas para seleccion de caracteristicas

Como enumerar todos los subconjuntos es inviable, se recurre a heuristicas. Meiri y Zahavi (2006)
subrayan que el problema es de optimizacion combinatoria, de tipo NP-duro, y sin solucion analitica.
Segun Xue et al. (2016), los algoritmos geneticos (GA) y la optimizacion por enjambre de particulas
(PSO) son las dos tecnicas de computacion evolutiva mas populares en este campo, con un aumento fuerte
del numero de publicaciones desde 2007, sobre todo de PSO. Agrawal et al. (2021) revisan una decada de
metaheuristicas (2009-2019) con variantes binarias, las clasifican segun su comportamiento e
identifican retos y lineas abiertas. Los cuatro metodos monoobjetivo de este trabajo tienen precedentes
directos:

- **Algoritmos geneticos.** Su representacion natural es una cadena binaria en la que un 1 indica que la
  variable se selecciona, y Siedlecki y Sklansky (1989) los introdujeron para este problema; Xue et al.
  (2016) los citan como uno de los trabajos mas tempranos, y Zhang y Sun (2002) recogen que obtuvieron
  buenos resultados.
- **PSO binario.** Kennedy y Eberhart (1997) adaptaron el PSO a variables binarias: la velocidad
  determina la probabilidad de que cada coordenada tome el valor 1, y, al contrario que en el PSO
  continuo, un limite de velocidad `v_max` menor permite una mayor tasa de "mutacion" (relevante para el
  parametro `v_max`, apartado 5.9). Xue et al. (2016) recogen ademas que el PSO continuo suele rendir
  mejor que el binario en seleccion multiobjetivo, porque en el binario la posicion se actualiza solo a
  partir de la velocidad.
- **Recocido simulado.** Kirkpatrick et al. (1983) lo propusieron por analogia con el recocido de
  solidos para problemas de optimizacion combinatoria; Meiri y Zahavi (2006) lo aplican a la
  seleccion de variables.
- **Busqueda tabu.** Fue introducida por Glover (1986, 1989). Zhang y Sun (2002) la aplican a la
  seleccion de caracteristicas con una cadena de ceros y unos, una vecindad que consiste en anadir o
  quitar una variable y, en cada iteracion, todo o una parte elegida al azar de la vecindad como conjunto
  de candidatos, que es el diseno de este trabajo (apartado 5.6). La comparan con un GA, con busquedas
  secuenciales (incluidas las flotantes SFFS y SBFS) y con ramificacion y poda, la encuentran
  prometedora por la calidad de los subconjuntos y su eficiencia, y estudian el efecto de sus parametros.
  Recomiendan una lista tabu de tamano entre `n/2` y `3n`; en nuestro ajuste (apartado 5.9) resultaron
  mejores tenencias cortas, aunque su lista y la nuestra no son equivalentes. Sus experimentos usan datos
  sinteticos (30, 60 y 100 variables) y los propios autores proponen probarlo con datos reales, que es lo
  que hace este trabajo.

#### 3.A.3 Enfoque multiobjetivo y NSGA-II

La seleccion de caracteristicas tiene dos objetivos, maximizar el rendimiento de clasificacion y
minimizar el numero de variables, que a menudo entran en conflicto, aunque no siempre: en algunos
subespacios quitar variables innecesarias tambien reduce el error (Xue et al., 2016). La mayoria de los
metodos, sin embargo, optimizan solo el rendimiento o combinan ambos objetivos en uno (Xue et al., 2016),
como hace la formulacion monoobjetivo de este trabajo con el parametro `alpha`. El enfoque de Pareto evita
esa combinacion y devuelve un conjunto de soluciones de compromiso. Hamdani et al. (2007) aplicaron
NSGA-II a la seleccion de caracteristicas con dos objetivos competidores, buscando soluciones
Pareto-optimas, con cinco bases de datos del repositorio UCI; segun Xue et al. (2016), la mayoria de los
GA multiobjetivo para este problema se basan en NSGA-II o en variantes. Xue et al. (2013) presentan lo
que describen como el primer estudio de PSO multiobjetivo para seleccion de caracteristicas, con dos
algoritmos que comparan con tres metodos evolutivos multiobjetivo conocidos: el primero obtiene
resultados comparables en la mayoria de los casos y el segundo mejora al primero y al resto.

NSGA-II (Deb et al., 2002) es el algoritmo que usamos: propone una ordenacion no dominada rapida con
coste `O(M N^2)` (frente a `O(M N^3)` en la version anterior), elitismo y un mecanismo de diversidad sin
parametro de comparticion, basado en la distancia de crowding. Nuestra implementacion se valido frente
a la de pymoo (Blank y Deb, 2020) en el apartado 5.10.

#### 3.A.4 Sobreajuste en la seleccion con wrapper

Un riesgo conocido del enfoque wrapper es el sobreajuste. Loughrey y Cunningham (2005) muestran que,
cuantos mas estados visita la busqueda, mayor es la probabilidad de encontrar un subconjunto con alta
precision interna pero mala generalizacion, y proponen un GA con parada temprana para reducirlo; Kohavi
y John (1997) ya discuten el problema, y Xue et al. (2016) senalan el sesgo de seleccion cuando se usan
todos los datos durante la busqueda. Esto justifica dos decisiones de este trabajo: separar una
particion de test que la busqueda nunca ve (apartado 4.7) y comparar en la discusion el error de
validacion cruzada con el error de test; ademas, el presupuesto limita la intensidad de la busqueda.

#### 3.A.5 Posicionamiento de este trabajo

Xue et al. (2016) senalan que faltan guias completas sobre las fortalezas y debilidades de los distintos
enfoques, y que el coste computacional, por el gran numero de evaluaciones, es una limitacion
importante de las tecnicas evolutivas; entre las vias para reducirlo mencionan los enfoques con modelos
sustitutos. Este trabajo compara, a pequena escala, cuatro metaheuristicas monoobjetivo (GA, BPSO, SA y
busqueda tabu) y NSGA-II bajo la misma funcion objetivo, el mismo clasificador, las mismas particiones y
el mismo presupuesto de evaluaciones, junto con metodos clasicos (seccion 5, parte B). Los modelos
sustitutos quedan como trabajo futuro.
<!-- REVISAR: ajusta esta ultima frase cuando Javier tenga los resultados finales y sepas que conclusiones saca la discusion. -->
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

### Referencias (parte A)

- Agrawal, P., Abutarboush, H. F., Ganesh, T. y Mohamed, A. W. (2021). Metaheuristic algorithms on
  feature selection: A survey of one decade of research (2009-2019). *IEEE Access*, 9, 26766-26791.
  https://doi.org/10.1109/ACCESS.2021.3056407
- Blank, J. y Deb, K. (2020). pymoo: Multi-objective optimization in Python. *IEEE Access*, 8, 89497-89509.
  https://doi.org/10.1109/ACCESS.2020.2990567
- Deb, K., Pratap, A., Agarwal, S. y Meyarivan, T. (2002). A fast and elitist multiobjective genetic
  algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
  https://doi.org/10.1109/4235.996017
- Glover, F. (1986). Future paths for integer programming and links to artificial intelligence.
  *Computers & Operations Research*, 13(5), 533-549. https://doi.org/10.1016/0305-0548(86)90048-1
- Glover, F. (1989). Tabu search - Part I. *ORSA Journal on Computing*, 1(3), 190-206.
  https://doi.org/10.1287/ijoc.1.3.190
- Hamdani, T. M., Won, J.-M., Alimi, A. M. y Karray, F. (2007). Multi-objective feature selection with
  NSGA II. En *Adaptive and Natural Computing Algorithms (ICANNGA 2007)*, Lecture Notes in Computer
  Science, vol. 4431, 240-247. Springer. https://doi.org/10.1007/978-3-540-71618-1_27
- Kennedy, J. y Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm. En
  *1997 IEEE International Conference on Systems, Man, and Cybernetics*, vol. 5, 4104-4108.
  https://doi.org/10.1109/ICSMC.1997.637339
- Kirkpatrick, S., Gelatt, C. D. y Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*,
  220(4598), 671-680. https://doi.org/10.1126/science.220.4598.671
- Kohavi, R. y John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*,
  97(1-2), 273-324. https://doi.org/10.1016/S0004-3702(97)00043-X
- Loughrey, J. y Cunningham, P. (2005). Overfitting in wrapper-based feature subset selection: The
  harder you try the worse it gets. En M. Bramer, F. Coenen y T. Allen (eds.), *Research and Development
  in Intelligent Systems XXI (AI-2004)*, 33-43. Springer. https://doi.org/10.1007/1-84628-102-4_3
- Meiri, R. y Zahavi, J. (2006). Using simulated annealing to optimize the feature selection problem in
  marketing applications. *European Journal of Operational Research*, 171(3), 842-858.
  https://doi.org/10.1016/j.ejor.2004.09.010
- Siedlecki, W. y Sklansky, J. (1989). A note on genetic algorithms for large-scale feature selection.
  *Pattern Recognition Letters*, 10(5), 335-347. https://doi.org/10.1016/0167-8655(89)90037-8
- Xue, B., Zhang, M. y Browne, W. N. (2013). Particle swarm optimization for feature selection in
  classification: A multi-objective approach. *IEEE Transactions on Cybernetics*, 43(6), 1656-1671.
  https://doi.org/10.1109/TSMCB.2012.2227469
- Xue, B., Zhang, M., Browne, W. N. y Yao, X. (2016). A survey on evolutionary computation approaches to
  feature selection. *IEEE Transactions on Evolutionary Computation*, 20(4), 606-626.
  https://doi.org/10.1109/TEVC.2015.2504420
- Zhang, H. y Sun, G. (2002). Feature selection using tabu search method. *Pattern Recognition*, 35(3),
  701-711. https://doi.org/10.1016/S0031-3203(01)00046-2
