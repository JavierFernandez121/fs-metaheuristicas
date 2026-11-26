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
  Recomiendan una lista tabu de tamano entre `n/2` y `3n`; en nuestro ajuste (apartado 5.9) las tenencias mas largas (15 y 30) fueron algo peores que la corta por defecto , aunque su lista y la nuestra no son equivalentes. Sus experimentos usan datos
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
# 3.1 El problema de seleccion de variables · B
Seleccionar variables persigue tres objetivos: mejorar la prediccion, obtener modelos mas rapidos y baratos de usar y entender mejor el proceso que genera los datos (Guyon y Elisseeff, 2003). Los metodos se agrupan en tres familias segun como se relacionan con el modelo: filtros, wrappers y metodos embebidos. Las tres aparecen en este trabajo: la informacion mutua es un filtro, todos los metodos de busqueda (SFS y las metaheuristicas) son wrappers y LASSO es embebido.

# 3.2 Filtros · B
Un filtro puntua las variables con una medida estadistica calculada solo con los datos, sin entrenar el modelo final, y se queda con las mejores. Son muy rapidos y no dependen del clasificador. La informacion mutua es una de las medidas mas usadas porque detecta cualquier tipo de dependencia, no solo la lineal; Battiti (1994) la propuso para elegir variables de forma voraz teniendo en cuenta tambien la redundancia con las ya elegidas, idea que Peng, Long y Ding (2005) formalizaron con el criterio mRMR (maxima relevancia, minima redundancia). La limitacion de los filtros univariantes es que evaluan cada variable por separado: no ven variables que solo son utiles en combinacion ni el sesgo del clasificador que se usara despues.

# 3.3 Wrappers · B
Kohavi y John (1997) formalizaron el enfoque wrapper: la seleccion es una busqueda en el espacio de subconjuntos y el propio algoritmo de aprendizaje, usado como caja negra, evalua cada candidato, normalmente con validacion cruzada. Su ventaja es que elige las variables que mejor funcionan con ese clasificador concreto; su inconveniente, el coste, porque cada evaluacion entrena modelos. Las estrategias de busqueda clasicas son voraces: la seleccion secuencial hacia delante (Whitney, 1971) y su version flotante (SFFS), que permite quitar variables ya anadidas (Pudil, Novovicova y Kittler, 1994). Las metaheuristicas son otra forma de recorrer el mismo espacio, con mas capacidad de escapar de optimos locales; Xue et al. (2016) revisan su uso en seleccion de variables (ver 3.x).

# 3.4 Metodos embebidos · B
En los metodos embebidos la seleccion forma parte del entrenamiento. El ejemplo de referencia es LASSO (Tibshirani, 1996): una penalizacion sobre la suma de los valores absolutos de los coeficientes lleva muchos de ellos exactamente a cero, de modo que el modelo ajustado ya indica que variables se usan. Es mucho mas barato que un wrapper, pero queda ligado a un modelo concreto (lineal, en el caso de LASSO).

# 3.5 Sobreajuste en la seleccion y comparacion rigurosa · B
Cuando una busqueda intensiva se guia por la estimacion de validacion cruzada, esa estimacion acaba sobreajustada: es optimista y no sirve para comparar metodos de seleccion entre si. Reunanen (2003) lo mostro empiricamente y concluyo que el rendimiento final debe medirse en un conjunto de test independiente; con ese criterio, algunas ventajas atribuidas a busquedas mas exhaustivas frente a otras mas simples desaparecian. Cawley y Talbot (2010) muestran que este sobreajuste en la seleccion de modelos puede ser tan grande como las diferencias entre algoritmos. Por eso este trabajo separa un 30 % de test que la busqueda no ve nunca (4.7) y analiza la diferencia entre error CV y error de test. Para comparar varios metodos se siguen las recomendaciones de Demsar (2006): test de Friedman para todos a la vez y test de rangos con signo de Wilcoxon por pares.

# 3.6 El NIPS 2003 Feature Selection Challenge · B
Madelon procede de este concurso, en el que se propusieron cinco conjuntos de datos de dominios distintos y se pedia clasificar con el menor numero de variables posible (Guyon et al., 2004). Participaron 78 grupos; las mejores propuestas combinaban filtros, wrappers o metodos embebidos con random forests, metodos de nucleo o redes neuronales. Al ser un problema con variables relevantes conocidas y mucho ruido, sigue siendo un banco de pruebas habitual para metodos de seleccion.

# 3.x Metaheuristicas y NSGA-II · A
<!-- Lo escribe Persona A -->

## 4. Formulacion matematica · A
# 4.1 Notacion y espacio de busqueda
Sea D_train = {(a_i, y_i)}, i = 1..m, el conjunto de entrenamiento (70 % de los datos, particion estratificada fija), con a_i en R^n. Una solucion es un vector binario x en {0,1}^n, donde x_j = 1 significa que la variable j entra en el modelo. Se escribe |x| = sum_j x_j para el numero de variables elegidas y se impone la restriccion |x| >= 1 (sin variables no hay modelo).
El espacio factible tiene 2^n - 1 soluciones: unas 1,07 * 10^9 para Breast Cancer (n = 30) y unas 3,3 * 10^150 para Madelon (n = 500). Enumerarlas es inviable, lo que justifica el uso de metaheuristicas.

# 4.2 Funcion de evaluacion (enfoque wrapper)
La calidad de x se mide entrenando un clasificador solo con las columnas j tales que x_j = 1:
err_CV(x) = 1 - (precision media en validacion cruzada estratificada de 5 particiones)
Clasificador: k-NN con k = 5, con estandarizacion de variables ajustada dentro de cada particion.
Las particiones de validacion cruzada son las mismas para todas las soluciones (semilla fija), de modo que err_CV es una funcion determinista de x.
Solo se usa el conjunto de entrenamiento. La particion de test (30 %) no interviene nunca en la busqueda: se usa unicamente al final para medir la precision del subconjunto elegido.

# 4.3 Version monoobjetivo (GA, BPSO, SA y busqueda tabu)
min  f(x) = alpha * err_CV(x) + (1 - alpha) * |x| / n      sujeto a |x| >= 1,  x en {0,1}^n
con alpha = 0,99. Con este valor, bajar un punto porcentual de error (0,01) mejora f en 0,0099, mientras que quitar una variable la mejora solo en 0,01 / n (0,00033 con n = 30; 0,00002 con n = 500). El termino de tamano actua, por tanto, sobre todo como criterio de desempate entre subconjuntos con el mismo error: prioriza la precision y, a igualdad de error, prefiere el subconjunto mas pequeño. 

# 4.4 Version biobjetivo (NSGA-II)
min  F(x) = ( err_CV(x), |x| )                              sujeto a |x| >= 1
Ya no hay un unico optimo: x domina a x' si err_CV(x) <= err_CV(x') y |x| <= |x'|, con al menos una desigualdad estricta. El frente de Pareto es el conjunto de soluciones no dominadas, y muestra el compromiso entre precision y tamano. Para comparar frentes se usa el hipervolumen: el area del plano (error, |x|/n) dominada por el frente respecto al punto de referencia (1, 1). Mayor hipervolumen es mejor.

# 4.5 Caracter del problema
Es un problema discreto y combinatorio, no convexo, sin gradiente, con una funcion objetivo costosa (cada evaluacion nueva entrena 5 modelos) y con muchas mesetas: subconjuntos vecinos distintos pueden dar exactamente el mismo error. Ademas err_CV es solo una estimacion del error de generalizacion.

# 4.6 Presupuesto y cache
Todos los metodos reciben el mismo presupuesto B = 3000 evaluaciones. Una evaluacion cuenta cuando se entrena un modelo nuevo; una mascara ya vista sale de la cache y no gasta presupuesto. Esto hace la comparacion justa, pero obliga a proteger los metodos frente a los que convergen y repiten soluciones (ver 5.2).

# 4.7 Sobreajuste de la seleccion
Como la busqueda minimiza err_CV sobre el mismo entrenamiento miles de veces, el subconjunto elegido tiende a quedar sobreajustado a esas particiones. Por eso se separa un test independiente, y la diferencia entre error CV y error de test se analiza en la discusion.

## 5. Metodos · A (metaheuristicas) · B (clasicos)
# 5.1 Marco comun
Todos los metodos cumplen la misma interfaz run(fitness, rng, **params) -> Result: reciben la funcion objetivo compartida (con cache y presupuesto) y un generador aleatorio con semilla, y devuelven la mejor mascara, su valor y la curva de convergencia. La representacion es una mascara binaria; tras cada operador se repara cualquier mascara vacia activando una variable al azar (restriccion |x| >= 1).
Salvo en el GA, las soluciones iniciales usan una densidad de unos distinta por individuo, tomada de forma uniforme en [0,02, 0,5] ([0,05, 0,5] en NSGA-II), para cubrir tamanos de subconjunto variados desde el principio; con una densidad fija de 0,5 todos tendrian cerca de n/2 variables. El GA usa por defecto 0,5 (opcion p_init=None para densidad variable; en el ajuste no mejoro).

# 5.2 Proteccion frente a la cache
Como las mascaras repetidas no gastan presupuesto, un metodo que converge puede proponer solo soluciones ya vistas y no agotar nunca B. Un guardian (_guard.py) detiene la busqueda tras 3000 evaluaciones consecutivas sin ningun entrenamiento nuevo y devuelve el mejor resultado hasta ese momento. En los experimentos con datos reales no llega a actuar; es una red de seguridad.

# 5.3 Algoritmo genetico (GA)
Idea: una poblacion de mascaras evoluciona por seleccion, cruce y mutacion.
P <- pop_size mascaras aleatorias; evaluar
mientras quede presupuesto:
    elite <- los 2 mejores de P (pasan sin reevaluarse)
    mientras falten individuos:
        p1, p2 <- torneo de tamano 3 (dos veces)
        h <- cruce uniforme de p1 y p2 con prob. 0,9 (si no, copia de p1)
        h <- mutacion bit-flip: cada bit se invierte con prob. 1/n
        reparar si h es vacia; evaluar h
    P <- elite + hijos
Parametros: pop_size = 30, cruce 0,9, mutacion 1/n, torneo k = 3, elitismo 2, p_init = 0,5.
Complejidad por generacion: O(pop_size * (n + C_eval)), donde C_eval es el coste de evaluar una mascara (validacion cruzada de un k-NN); C_eval domina.
Por que: es la metaheuristica poblacional de referencia y la mutacion 1/n cambia de media un bit por hijo, adecuado para busqueda local en {0,1}^n.
# 5.4 PSO binario (BPSO)
Idea (Kennedy y Eberhart, 1997): cada particula tiene una mascara x y una velocidad real v por variable; la velocidad se actualiza como en el PSO continuo y la posicion se vuelve a muestrear con una funcion de transferencia sigmoide.
inicializar particulas, velocidades, pbest y gbest
mientras quede presupuesto:
    para cada particula i:
        v <- w*v + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i);  v <- clip(v, -v_max, v_max)
        x_i[j] <- 1 con probabilidad 1 / (1 + exp(-v[j]))
        reparar; evaluar; actualizar pbest_i y gbest
Parametros: n_particles = 30, w = 0,9, c1 = c2 = 1,5, v_max = 4.
Complejidad por iteracion: O(n_particles * (n + C_eval)).
Por que: es el metodo de enjambre mas usado en seleccion de variables; limitar v_max evita que la probabilidad llegue a 0 o 1 y mantiene algo de exploracion (sigmoide(4) = 0,982).

# 5.5 Recocido simulado (SA)
Idea (Kirkpatrick et al., 1983): un solo candidato que se mueve por vecinos e incluso acepta empeoramientos con probabilidad exp(-delta / T), que decrece al enfriarse T.
x <- mascara aleatoria
T0 <- media de |delta f| de 20 vecinos de x     (calibracion automatica; gasta presupuesto)
mientras quede presupuesto:
    repetir 20 veces:
        y <- x con entre 1 y k_max bits invertidos; reparar; evaluar
        si f(y) <= f(x) o aleatorio < exp(-(f(y)-f(x))/T):  x <- y
    T <- 0,95 * T
Parametros: T0 calibrada, cooling = 0,95, iters_per_temp = 20, k_max = 5.
Complejidad por iteracion: O(C_eval).
Por que: es el metodo de trayectoria mas simple y sirve de contraste con los poblacionales; la calibracion de T0 evita depender de la escala de f.

# 5.6 Busqueda tabu
Idea (Glover, 1986): busqueda local que se mueve siempre al mejor vecino admisible, aunque empeore, y prohibe temporalmente deshacer movimientos recientes para no volver sobre sus pasos.
x <- mascara aleatoria
mientras quede presupuesto:
    vecinos <- 20 bits j al azar; para cada uno, y <- x con el bit j invertido; evaluar
    descartar los tabu que no superen al mejor global (criterio de aspiracion)
    mover al mejor restante; el bit j queda tabu durante `tenure` iteraciones
    si pasan 30 iteraciones sin mejorar el mejor global:
        perturbar el 5 % de los bits y vaciar la lista tabu
Parametros: n_neighbors = 20, tenure = max(5, round(sqrt(n))) (5 con n = 30; 22 con n = 500), patience = 30, perturbacion 5 %.
Complejidad por iteracion: O(n_neighbors * C_eval).
Por que: con n = 500, evaluar los 500 vecinos gastaria el presupuesto en 6 iteraciones, por eso se muestrea una parte; la diversificacion evita estancarse en un optimo local.
# 5.7 NSGA-II
Idea (Deb et al., 2002): algoritmo genetico multiobjetivo que ordena la poblacion por frentes de no dominancia y, dentro de un frente, prefiere las soluciones menos aglomeradas (distancia de crowding).
P <- pop_size mascaras aleatorias; evaluar (err_CV, |x|)
mientras quede presupuesto:
    Q <- hijos por torneo binario (menor rango; a igualdad, mayor crowding),
         cruce uniforme (prob. 0,9) y mutacion bit-flip (prob. 1/n)
    evaluar Q
    R <- P + Q;  frentes <- ordenacion no dominada de R
    P <- frentes completos mientras quepan; el ultimo se corta por mayor crowding
devolver el primer frente de P
Parametros: pop_size = 40, cruce 0,9, mutacion 1/n.
Complejidad por generacion: O(M * N^2) para la ordenacion (M = 2 objetivos, N = 2 * pop_size) mas las evaluaciones; la ordenacion se implemento de forma vectorizada.
Para que la curva de convergencia y el mejor f sean comparables con los demas metodos, cada individuo se evalua tambien con la funcion monoobjetivo (sale de la cache, no gasta presupuesto).

# 5.8 Resumen de parametros finales
Metodo	Parametros finales
GA	pop_size = 30, cruce 0,9, mutacion 1/n, torneo 3, elitismo 2, p_init = 0,5
BPSO	n_particles = 30, w = 0,9, c1 = c2 = 1,5, v_max = 4
SA	T0 calibrada, cooling = 0,95, iters_per_temp = 20, k_max = 5
Tabu	n_neighbors = 20, tenure = max(5, round(sqrt(n))), patience = 30, perturbacion 5 %
NSGA-II	pop_size = 40, cruce 0,9, mutacion 1/n

# 5.9 Ajuste de parametros
Protocolo. Solo con el 70 % de entrenamiento de Breast Cancer (398 muestras, 30 variables), presupuesto 3000 y 5 semillas por configuracion (3 en BPSO). Se partio de la configuracion inicial y se vario un parametro cada vez (script herramientas_a/ajuste_parametros.py). Se decidio cambiar un valor solo si la mejora de f superaba unas dos veces el error estandar de la media; si no, se considero ruido. El test no se uso.
Resultados (media de f, menor es mejor; entre parentesis, error estandar):
Metodo	Configuracion inicial	Cambio probado	Resultado	Decision
BPSO	w = 0,7: 0,0145 (0,0004)	w = 0,9	0,0105 (0,0002)	adoptar
BPSO		w = 0,4	0,0179 (0,0007)	peor
SA	k_max = 3: 0,0120 (0,0007)	k_max = 5	0,0107 (0,0001)	adoptar
SA		cooling = 0,99	0,0151 (0,0012)	peor: enfria demasiado despacio para este presupuesto
GA	0,0115 (0,0006)	mejor: pop_size = 50	0,0107 (0,0003)	ruido; se mantiene
| Tabu | 0,0103 (0,0001) | `tenure = 15` / `tenure = 30` | 0,0112 (0,0004) / 0,0109 (0,0003) | algo peores; se mantiene el valor automatico |
NSGA-II	0,0112 (0,0004)	todas	0,0111 a 0,0116	ruido; se mantiene
En tabu, todas las configuraciones alcanzaron el mismo error de validacion cruzada (0,0075) salvo `tenure = 15` (0,0080); lo demas cambia, a lo sumo, en una variable. Estos datos son posteriores a corregir un error de la aspiracion y de la diversificacion (ver apartado de IA); antes del arreglo la configuracion inicial daba 0,0106.
Limitaciones. (1) Breast Cancer es un problema facil y muchas configuraciones saturan en el mismo error, por lo que discrimina poco: conviene comprobar los dos cambios adoptados en Madelon. (2) Variar un parametro cada vez no detecta interacciones. (3) BPSO se probo con solo 3 semillas, asi que su error estandar es poco fiable. (4) Con unas 40 configuraciones probadas, alguna mejora limite puede deberse al azar, por eso solo se adoptaron las claramente por encima del ruido.

**Comprobacion en un conjunto sintetico tipo Madelon** (600 muestras, 200 variables: 5 informativas
y 15 redundantes; 420 de entrenamiento; presupuesto 1500, 3 semillas). BPSO con `w = 0,9` obtiene
f = 0,1320 (error estandar 0,0047) frente a 0,1417 (0,0035) con `w = 0,7`: la mejora (0,0097) es
unas dos veces el error estandar, en el limite del criterio, pero con el mismo signo que en
Breast Cancer. En SA, `k_max = 5` (0,1074; 0,0056) no mejora a `k_max = 3` (0,1056; 0,0023);
`k_max = 1` es el mejor en valor medio (0,0920) pero con mucha variabilidad (error estandar 0,0105),
y `cooling = 0,99` vuelve a ser claramente peor (0,1315; 0,0039). Por tanto, `w = 0,9` se
confirma en ambos conjuntos, mientras que `k_max = 5` solo es claro en Breast Cancer y su efecto
parece depender del numero de variables.

# 5.10 Validacion de NSGA-II frente a pymoo
Para validar la implementacion propia se comparo con NSGA2 de pymoo 0.6.2 usando la misma funcion objetivo, el mismo presupuesto (3000) y la misma poblacion (40), con 5 semillas. En ambos casos el frente es el conjunto no dominado de todas las mascaras evaluadas, y la metrica es el hipervolumen normalizado (referencia (1, 1)):
Semilla	HV propio	HV pymoo	Cociente
0	0,9526	0,9545	0,998
1	0,9545	0,9544	1,000
2	0,9545	0,9545	1,000
3	0,9525	0,9543	0,998
4	0,9527	0,9542	0,998
Media	0,9533 (± 0,0009)	0,9544 (± 0,0001)	0,999
Los frentes son practicamente equivalentes (diferencia media del 0,1 %). La pequena ventaja de pymoo en algunas semillas es compatible con la diferencia de inicializacion: pymoo activa cada bit con probabilidad 0,5 y la implementacion propia usa una densidad variable en [0,05, 0,5].
# 5.11 Metodos clasicos y lineas base · B
Para saber si las metaheuristicas aportan algo hay que compararlas con metodos mas sencillos. Se usan cinco referencias, todas con la misma interfaz run(fitness, rng) -> Result y la misma funcion objetivo f(x) de 4.3: cada metodo elige su mejor configuracion con el mismo criterio que las metaheuristicas y el resultado se mide igual (error CV, precision en test y numero de variables).

Todas las variables. La mascara x = (1, ..., 1). Es la referencia trivial: un metodo de seleccion que no la mejore no esta eliminando ruido.

Busqueda aleatoria. Evalua mascaras aleatorias hasta agotar el presupuesto; la densidad de unos se sortea en cada iteracion en [0,02, 0,6] para cubrir tamanos variados. Usa el mismo guardian que las metaheuristicas (5.2). Es la linea base minima: si una metaheuristica no le gana con el mismo presupuesto, no aporta nada.

Seleccion secuencial hacia delante (SFS) (Whitney, 1971). Wrapper voraz: parte del conjunto vacio y en cada paso anade la variable que mas reduce f; para cuando ninguna mejora o cuando se agota el presupuesto.
S <- vacio; f_S <- infinito
repetir:
    para cada j que no esta en S: evaluar f(S + {j})
    j* <- la de menor f
    si f(S + {j*}) >= f_S: parar
    S <- S + {j*};  f_S <- f(S + {j*})
Se implementa sobre la funcion objetivo comun (no con la clase de scikit-learn) para que gaste el mismo presupuesto y use la misma cache. Coste: el paso k evalua n - k + 1 mascaras, O(n^2) evaluaciones para recorrer el camino completo. Con Madelon (n = 500) el presupuesto de 3000 solo alcanza para seis pasos completos (500 + 499 + ... + 495 = 2985 evaluaciones), asi que SFS no puede pasar de 6 variables: es una limitacion del metodo en alta dimension, no de la implementacion. Ademas sufre el efecto nesting: una variable anadida ya no se puede quitar; los metodos flotantes (Pudil et al., 1994) lo corrigen, pero no se incluyen.

Filtro por informacion mutua (MI) (Battiti, 1994). Puntua cada variable por separado con su informacion mutua con la clase, estimada por vecinos proximos (mutual_info_classif de scikit-learn, basado en Ross, 2014). Ordena las variables y prueba las k primeras para k en {1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300}; devuelve el k con menor f. Gasta como mucho 15 evaluaciones. Es univariante: no detecta redundancia (dos variables casi iguales puntuan igual de alto) ni interacciones (una variable inutil por si sola puede ser clave junto con otra). El estimador tiene una pequena componente aleatoria, por eso se ejecuta con todas las semillas.

LASSO (Tibshirani, 1996). Metodo embebido: la seleccion ocurre al entrenar. Se usa su version para clasificacion, regresion logistica con penalizacion L1 (solver liblinear), sobre variables estandarizadas. La penalizacion lleva a cero los coeficientes de las variables que no compensan; las que quedan con coeficiente no nulo forman la mascara. Se recorren 20 valores de C en escala logaritmica entre 10^-3 y 10 (C pequeno = mas penalizacion = menos variables) y se devuelve la mascara con menor f. Gasta como mucho 20 evaluaciones. Es lineal: puede fallar cuando la clase depende de las variables de forma no lineal, como en Madelon.

Presupuesto. Busqueda aleatoria y SFS pueden gastar las 3000 evaluaciones; MI y LASSO, como mucho 15 y 20, porque su espacio de busqueda es mucho mas pequeno. Se informa siempre del numero real de evaluaciones de cada metodo para que la diferencia de coste sea explicita. Todas las variables, SFS y LASSO son deterministas y se ejecutan una sola vez.

## 6. Implementacion  ·  B
# 6.1 Organizacion del codigo
El codigo esta en Python 3 (numpy, scipy, pandas, scikit-learn, matplotlib, joblib) y se organiza en un paquete y unos scripts:
Modulo	Contenido
src/fsmeta/data.py	Carga de datasets y particion train/test fija
src/fsmeta/fitness.py	Funcion objetivo comun: wrapper con k-NN, validacion cruzada, cache y presupuesto
src/fsmeta/metaheuristics/	GA, BPSO, SA, tabu, NSGA-II, busqueda aleatoria, interfaz comun (base.py) y guardian (_guard.py)
src/fsmeta/baselines.py	Todas las variables, SFS, informacion mutua y LASSO
src/fsmeta/stats.py	Friedman, Wilcoxon con correccion de Holm, frente de Pareto e hipervolumen 2D
experiments/	config.py (toda la configuracion), run_all.py, make_tables.py, make_figures.py
tests/	Pruebas automaticas (pytest)
El nucleo de todos los metodos de busqueda es implementacion propia. Las librerias se usan solo para el clasificador (k-NN y regresion logistica de scikit-learn), el estimador de informacion mutua, los tests estadisticos (scipy) y la validacion de NSGA-II (pymoo, 5.10).

# 6.2 Decisiones de diseno
Una sola funcion objetivo para todos. La clase FitnessFunction concentra todo lo que hace justa la comparacion: el mismo clasificador (estandarizacion + k-NN con k = 5 dentro de un pipeline, de modo que la estandarizacion se ajusta solo con la parte de entrenamiento de cada pliegue), las mismas 5 particiones de validacion cruzada estratificada (semilla fija) y el mismo presupuesto. Cada metodo solo usa dos llamadas: fitness(mask), que devuelve f(x), y fitness.multi(mask), que devuelve (err_CV(x), |x|) para NSGA-II.
Cache. Cada mascara se guarda comprimida (np.packbits) como clave de un diccionario con su error. Una mascara repetida no se reentrena y no gasta presupuesto. Esto ahorra mucho tiempo en los metodos que convergen, y obliga a usar el guardian de 5.2.
Presupuesto. n_evals cuenta solo entrenamientos nuevos. Al llegar al presupuesto, la funcion lanza la excepcion BudgetExhausted; cada metodo la captura y devuelve su mejor solucion. Ningun metodo puede pasarse y ninguno tiene que llevar la cuenta por su cuenta.
Mejor solucion y convergencia. La propia funcion objetivo guarda la mejor mascara vista y la curva [(n_evals, mejor f)], de modo que todas las curvas de convergencia se miden en la misma escala (evaluaciones) y se construyen igual para todos los metodos.
Reproducibilidad. La particion train/test usa la semilla 42 y las particiones de validacion cruzada la semilla 0; cada ejecucion recibe su propio generador np.random.default_rng(semilla) y ningun metodo usa el azar global de numpy. Con la misma semilla se obtiene exactamente el mismo resultado (lo comprueba una prueba automatica).
Separacion del test. La particion de test nunca se pasa a la funcion objetivo. Solo run_all.py la usa, al final de cada ejecucion: entrena el clasificador con todo el entrenamiento y las variables elegidas y mide la precision en test.

# 6.3 Flujo de los experimentos
run_all.py lee config.py (datasets, metodos, semillas, presupuesto, parametros), ejecuta cada par (metodo, semilla) y guarda una fila por ejecucion en results/<dataset>.csv con el error CV, f, la precision en test, el numero de variables, las evaluaciones gastadas, el tiempo y las variables elegidas. La curva de convergencia (y el frente de Pareto en NSGA-II) se guarda en results/histories/. Como las ejecuciones son independientes, se lanzan en paralelo con joblib (opcion --jobs). make_tables.py genera las tablas (media y desviacion tipica) y los tests estadisticos, y make_figures.py las figuras: ninguna cifra de la memoria se escribe a mano.

# 6.4 Pruebas
tests/test_basics.py comprueba que la particion es fija, que la cache no gasta presupuesto, que ningun metodo se pasa del presupuesto, que todos terminan cuando el espacio de busqueda es mas pequeno que el presupuesto (6 variables, 63 mascaras) y que el frente de Pareto y el hipervolumen coinciden con un ejemplo calculado a mano. tests/test_persona_a.py comprueba los operadores de NSGA-II frente a un calculo por fuerza bruta, que el frente devuelto es no dominado, la reproducibilidad y que los metodos encuentran las variables informativas de un problema sintetico.

# 6.5 Dificultades encontradas
Bucles infinitos por la cache. Como una mascara repetida no gasta presupuesto, un metodo que solo propone mascaras ya vistas no termina nunca. Pasaba en las metaheuristicas al converger y en la busqueda aleatoria cuando hay menos combinaciones posibles que presupuesto. Se resolvio con un guardian comun (5.2) y una prueba automatica que lo detecta.
Coste de la evaluacion. Cada evaluacion entrena 5 modelos. En un problema del tamano de Madelon cuesta unos 50 ms, de modo que una ejecucion de 3000 evaluaciones tarda unos minutos y el experimento completo, horas. Se mitigo con la cache, un clasificador barato (k-NN) y la ejecucion en paralelo.
NSGA-II y la comparabilidad. fitness.multi no actualiza el mejor f ni la curva de convergencia, por lo que NSGA-II evalua tambien f(x) en cada individuo (sale de la cache, sin coste), ver 5.7.
Cambio de interfaz de scikit-learn. En la version 1.8 el parametro penalty de LogisticRegression pasa a estar obsoleto en favor de l1_ratio; baselines.py elige la sintaxis segun la version instalada para funcionar en ambas.

## 7. Datos  ·  B
# 7.1 Conjuntos utilizados
Dataset	Muestras	Variables	Clases	Uso	Licencia
Breast Cancer Wisconsin (Diagnostic)	569	30	212 malignos / 357 benignos	Depuracion y ajuste de parametros	CC BY 4.0
Madelon	2600 (2000 train + 600 valid)	500	1300 / 1300	Dataset principal	CC BY 4.0

Breast Cancer Wisconsin (Diagnostic) (Wolberg et al., 1993; UCI, doi:10.24432/C5DW2B). Cada muestra describe los nucleos celulares de la imagen de una puncion de una masa mamaria con 30 variables continuas (media, error estandar y peor valor de 10 medidas, como radio, textura o concavidad). Se carga desde scikit-learn (load_breast_cancer), que incluye una copia. Es un problema facil (con todas las variables, k-NN ya acierta en torno al 96 % en test) y pequeno (2^30 subconjuntos), por eso se usa para depurar y ajustar parametros, no para sacar conclusiones.

Madelon (Guyon, 2004; UCI, doi:10.24432/C5602H). Conjunto artificial del NIPS 2003 Feature Selection Challenge (Guyon et al., 2004). Los puntos se agrupan en 32 grupos situados en los vertices de un hipercubo de dimension 5 y etiquetados al azar con +1 o -1; esas 5 dimensiones son las variables informativas. Se anaden 15 combinaciones lineales suyas (20 variables relevantes en total) y 480 variables de ruido sin poder predictivo ("probes"), y se desordenan variables y muestras. Las variables toman valores enteros. Es un caso de prueba ideal: se sabe que solo 20 de las 500 variables sirven, el problema es no lineal y las variables utiles apenas destacan una a una, lo que perjudica a los filtros univariantes y a los modelos lineales.

# 7.2 Origen y particiones
Las etiquetas del test oficial del NIPS 2003 no son publicas, asi que se usan las particiones etiquetadas (train y valid, 2600 muestras), igual que la version de OpenML (id 1485). data.py lee los archivos originales de UCI si estan en data/raw/ (el zip de la pagina del dataset) y, si no, descarga la version de OpenML. Las etiquetas -1/+1 se pasan a 0/1.
Cada conjunto se divide una sola vez en 70 % de entrenamiento y 30 % de test, de forma estratificada (misma proporcion de clases en ambas partes) y con semilla fija 42:
Dataset	Entrenamiento	Test
Breast Cancer	398 (148 malignos / 250 benignos)	171 (64 / 107)
Madelon	1820 (910 / 910)	780 (390 / 390)
El entrenamiento es lo unico que ve la busqueda: dentro de el se hace la validacion cruzada de 5 particiones de la funcion objetivo (4.2). El test solo se usa en la medicion final.

# 7.3 Preprocesado
No se elimina ni se transforma ninguna variable de antemano, porque seleccionarlas es justamente el problema. El unico preprocesado es la estandarizacion (media 0, desviacion 1), necesaria porque k-NN usa distancias; va dentro del pipeline del clasificador, de modo que en cada pliegue se ajusta solo con los datos de entrenamiento de ese pliegue y no hay fuga de informacion. Ninguno de los dos conjuntos tiene valores ausentes.

## 8. Experimentacion y resultados  ·  B
# 8.1 Diseno experimental
Cada metodo se ejecuta sobre la particion de entrenamiento de cada conjunto con la funcion objetivo de 4.3 (k-NN con k = 5, validacion cruzada estratificada de 5 particiones, alpha = 0,99) y el mismo presupuesto de B = 3000 evaluaciones. Los metodos estocasticos (GA, BPSO, SA, tabu, NSGA-II, busqueda aleatoria e informacion mutua) se ejecutan con 10 semillas (0 a 9); los deterministas (todas las variables, SFS y LASSO), una vez. De cada ejecucion se guarda:
- error CV del subconjunto elegido (lo que optimiza la busqueda);
- precision en test: el k-NN se entrena con todo el entrenamiento y las variables elegidas y se evalua en el 30 % de test, que la busqueda no ha visto (lo que importa);
- numero de variables, evaluaciones gastadas y tiempo;
- brecha de sobreajuste: error en test menos error CV.
Para comparar metodos se usan el test de Friedman (todos a la vez) y el de Wilcoxon por pares con correccion de Holm, emparejando por semilla (Demsar, 2006), sobre dos metricas: el error CV (calidad de la busqueda) y la precision en test (generalizacion). Con 10 semillas, el menor p-valor exacto posible de Wilcoxon es 0,002; tras corregir por 21 comparaciones queda en 0,041, asi que solo se detectan como significativas las diferencias que se repiten en las 10 semillas.
Todas las tablas salen de results/tables_<dataset>.md y las figuras de figures/, generadas por experiments/make_tables.py y experiments/make_figures.py.

# 8.2 Breast Cancer
Metodo	Precision test	Error CV	Brecha	Variables	Evaluaciones
LASSO	0,9649	0,0352	-0,0001	4	13
GA	0,9632 ± 0,0078	0,0188 ± 0,0013	0,0180 ± 0,0082	12,7 ± 2,7	3000
NSGA-II	0,9626 ± 0,0063	0,0191 ± 0,0017	0,0183 ± 0,0066	6,1 ± 1,1	3000
BPSO	0,9614 ± 0,0111	0,0193 ± 0,0012	0,0193 ± 0,0113	13,4 ± 3,3	3000
Tabu	0,9608 ± 0,0078	0,0178 ± 0,0014	0,0213 ± 0,0072	10,3 ± 2,6	3000
Info. mutua	0,9591 ± 0,0000	0,0351 ± 0,0000	0,0058 ± 0,0000	30	9
Todas	0,9591	0,0351	0,0058	30	1
SA	0,9538 ± 0,0136	0,0188 ± 0,0013	0,0274 ± 0,0129	12,3 ± 2,4	3000
Aleatoria	0,9520 ± 0,0077	0,0241 ± 0,0013	0,0239 ± 0,0072	12,8 ± 3,0	3000
SFS	0,9474	0,0226	0,0301	6	189
(media ± desviacion tipica sobre 10 semillas; NSGA-II: la solucion de menor f de cada ejecucion)

Calidad de la busqueda. Las cinco metaheuristicas bajan el error CV a 0,018-0,019, frente a 0,024 de la busqueda aleatoria con el mismo presupuesto y 0,035 con todas las variables (Friedman: chi2 = 41,7, p = 2·10^-7). Las diferencias con la busqueda aleatoria y con informacion mutua son significativas tras Holm para todas ellas (p_holm = 0,041, el minimo posible); entre metaheuristicas no hay diferencias significativas. La busqueda tabu obtiene el menor error CV medio (0,0178) y la curva de convergencia (figures/convergence_breast_cancer.png) muestra que es tambien la que mas rapido baja: la mediana de tabu alcanza f = 0,0266 a las 150 evaluaciones, valor al que la mediana del GA no llega hasta unas 300. NSGA-II arranca mas despacio pero a partir de unas 600 evaluaciones tiene la mejor mediana de f, empatada al final con tabu (0,0208).
Generalizacion. En test la imagen cambia: las precisiones quedan entre 0,952 y 0,965 y, aunque Friedman detecta diferencias (chi2 = 13,7, p = 0,033), ningun par es significativo tras la correccion de Holm. LASSO, con solo 4 variables y 13 evaluaciones, obtiene la mejor precision en test (0,965) pese a tener el peor error CV.
Sobreajuste de la seleccion. Todos los wrappers tienen una brecha positiva de 0,018 a 0,030: el error en test es unas dos veces el error CV que creian haber conseguido. Con todas las variables la brecha es 0,006, y con LASSO, practicamente cero. Es exactamente el efecto descrito por Reunanen (2003): cuanto mas intensamente se optimiza la estimacion CV, mas optimista se vuelve. En Breast Cancer, el problema es tan facil que el margen de mejora real es pequeno (todas las variables ya dan 0,959) y el sobreajuste se come la ventaja de la busqueda (figures/boxplot_breast_cancer.png).
Tamano del subconjunto. NSGA-II consigue el mismo error CV que el GA con la mitad de variables (6,1 frente a 12,7), porque su segundo objetivo empuja explicitamente hacia subconjuntos pequenos, mientras que en la funcion monoobjetivo el termino de tamano (1 - alpha = 0,01) solo desempata. Su frente (figures/pareto_breast_cancer.png) muestra que con 3 variables ya se baja de un 3 % de error CV y que a partir de 5 la mejora es marginal. El hipervolumen medio del frente final es 0,9445 ± 0,0014; es algo menor que el de la validacion de 5.10 (0,953) porque aqui se calcula sobre el frente de la poblacion final y alli sobre todas las mascaras evaluadas.

# 8.3 Madelon
Metodo	Precision test	Error CV	Brecha	Variables	Relevantes elegidas (de 20)	Evaluaciones
SFS	0,8962	0,1022	0,0016	6	6	3000
NSGA-II	0,8959 ± 0,0131	0,1084 ± 0,0166	-0,0042 ± 0,0085	7,8 ± 3,0	7,4 ± 2,5	3000
LASSO	0,6641	0,3429	-0,0070	3	3	16
Info. mutua	0,6294 ± 0,0353	0,3588 ± 0,0367	0,0118 ± 0,0101	14,6 ± 21,7	4,5 ± 2,1	15
GA	0,5864 ± 0,0170	0,3294 ± 0,0137	0,0842 ± 0,0147	241,8 ± 7,6	18,4 ± 0,8	3000
Aleatoria	0,5760 ± 0,0123	0,4104 ± 0,0081	0,0135 ± 0,0151	136,2 ± 105,0	10,1 ± 5,9	3000
BPSO	0,5691 ± 0,0151	0,3815 ± 0,0057	0,0494 ± 0,0176	242,8 ± 12,0	17,1 ± 1,4	3000
SA	0,5673 ± 0,0207	0,3533 ± 0,0128	0,0794 ± 0,0233	232,7 ± 15,9	16,3 ± 2,2	3000
Tabu	0,5665 ± 0,0264	0,3626 ± 0,0166	0,0709 ± 0,0165	182,6 ± 45,4	14,2 ± 2,3	3000
Todas	0,5090	0,4434	0,0476	500	20	1
(media ± desviacion tipica sobre 10 semillas; las 20 variables relevantes se identifican en el entrenamiento porque son las unicas correladas entre si: su maxima |correlacion| con otra variable es al menos 0,67, y la de las 480 de ruido no pasa de 0,12)

En Madelon los metodos se separan en dos grupos muy distintos:
NSGA-II y SFS resuelven el problema. Eligen entre 5 y 14 variables, casi todas relevantes (el 96 % en NSGA-II y el 100 % en SFS), y llegan a una precision en test de 0,896, frente a 0,509 con todas las variables (una mejora de 39 puntos). NSGA-II es significativamente mejor que todas las demas metaheuristicas, la busqueda aleatoria y la informacion mutua, tanto en error CV como en precision en test (Wilcoxon con Holm, p_holm = 0,041 en todos los pares). Su brecha de sobreajuste es practicamente nula (-0,004).
Las cuatro metaheuristicas monoobjetivo fracasan. GA, BPSO, SA y tabu terminan con 180-240 variables: conservan casi todas las relevantes (de 14 a 18 de 20), pero tambien unas 200 de ruido. Su precision en test (0,567-0,586) no es significativamente distinta de la de la busqueda aleatoria, y su brecha es la mayor de todos los metodos (0,05-0,08): con tantas variables de ruido, la validacion cruzada premia combinaciones que no se sostienen en test. Sus curvas de convergencia (figures/convergence_madelon.png) se aplanan pronto en f ≈ 0,33-0,38, mientras que la de NSGA-II sigue bajando hasta 0,10.
Los metodos rapidos tambien fallan, por motivos conocidos. La informacion mutua (0,629) puntua cada variable por separado, y en Madelon las variables relevantes apenas destacan solas porque la clase depende de combinaciones de ellas (7.1); solo recupera 4,5 relevantes de media, y la mitad de lo que elige es ruido. LASSO (0,664) es lineal y el problema no lo es: elige 3 variables, todas relevantes, pero no puede usar las combinaciones no lineales que las hacen utiles.

Por que fracasan las monoobjetivo. Con alpha = 0,99, quitar una variable de ruido mejora f en (1 - alpha)/n = 0,00002, cifra muy inferior a la variacion del error CV entre subconjuntos vecinos. Partiendo de mascaras con decenas o cientos de variables, el paisaje es una gran meseta: invertir uno o pocos bits casi no cambia el error, porque con k-NN la distancia sigue dominada por los cientos de variables de ruido que quedan. Solo cuando quedan pocas variables de ruido el error cae bruscamente, y las metaheuristicas no llegan a esa zona con 3000 evaluaciones. NSGA-II si llega porque su segundo objetivo empuja de forma explicita hacia subconjuntos pequenos y la distancia de crowding mantiene en la poblacion soluciones de todos los tamanos: los individuos pequenos con variables relevantes sobreviven aunque su error inicial sea alto, y se recombinan entre si. SFS llega porque construye el subconjunto desde el vacio, aunque el presupuesto solo le permite 6 pasos (5.11). Esto apunta a una limitacion de la formulacion monoobjetivo de 4.3 en alta dimension, mas que de los algoritmos: un peso mayor para el termino de tamano o una inicializacion mucho mas dispersa son las alternativas naturales (trabajo futuro).

El frente de Pareto revela la estructura del problema. Agregando los frentes de las 10 semillas (figures/pareto_madelon.png), el error CV medio cae de 0,46 con 1 variable a 0,32 con 3, 0,21 con 4 y 0,13 con 5, y a partir de ahi se estanca (0,12 con 6-7 variables) o empeora ligeramente. El codo esta en 5 variables, justo la dimension intrinseca de Madelon (5 variables informativas; las otras 15 relevantes son combinaciones lineales suyas y no aportan informacion nueva). La precision en test sigue el mismo perfil (0,88-0,89 a partir de 5 variables). El hipervolumen medio es 0,884 ± 0,018, con dos semillas de diez (0 y 6) algo por detras (0,855 y 0,851) por no completar el frente en la zona de 5-6 variables.

Coste. NSGA-II es tambien el mas rapido de los metodos con presupuesto completo (99 s por ejecucion, frente a 160-200 s de las otras metaheuristicas), porque evaluar subconjuntos pequenos es mas barato.

# 8.4 Resumen de resultados
- En un problema facil y de dimension baja (Breast Cancer) todas las metaheuristicas optimizan mucho mejor que la busqueda aleatoria, pero la ventaja desaparece en test por el sobreajuste de la seleccion; un metodo barato como LASSO generaliza igual o mejor.
- En un problema dificil y de alta dimension (Madelon), la formulacion importa mas que el algoritmo: la version biobjetivo (NSGA-II) encuentra las variables relevantes y pasa de 0,51 a 0,90 de precision en test, mientras que las cuatro monoobjetivo con alpha = 0,99 no mejoran a la busqueda aleatoria.
- La brecha entre error CV y error de test crece con el numero de variables elegidas: casi nula para los subconjuntos pequenos de NSGA-II y SFS, de 0,05 a 0,08 para los de 200 variables.
- El frente de Pareto de NSGA-II recupera la dimension intrinseca de Madelon (5 variables).

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
# Referencias de Persona B (comprobadas: DOI o enlace)
Battiti, R. (1994). Using mutual information for selecting features in supervised neural net learning. IEEE Transactions on Neural Networks, 5(4), 537-550. https://doi.org/10.1109/72.298224
Cawley, G. C. y Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html
Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30. https://jmlr.org/papers/v7/demsar06a.html
Guyon, I. (2004). Madelon [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5602H
Guyon, I. y Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182. https://jmlr.org/papers/v3/guyon03a.html
Guyon, I., Gunn, S., Ben-Hur, A. y Dror, G. (2004). Result analysis of the NIPS 2003 feature selection challenge. Advances in Neural Information Processing Systems 17. https://proceedings.neurips.cc/paper/2004/hash/5e751896e527c862bf67251a474b3819-Abstract.html
Kohavi, R. y John, G. H. (1997). Wrappers for feature subset selection. Artificial Intelligence, 97(1-2), 273-324. https://doi.org/10.1016/S0004-3702(97)00043-X
Peng, H., Long, F. y Ding, C. (2005). Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8), 1226-1238. https://ieeexplore.ieee.org/document/1453511
Pudil, P., Novovicova, J. y Kittler, J. (1994). Floating search methods in feature selection. Pattern Recognition Letters, 15(11), 1119-1125. https://doi.org/10.1016/0167-8655(94)90127-9
Reunanen, J. (2003). Overfitting in making comparisons between variable selection methods. Journal of Machine Learning Research, 3, 1371-1382. https://jmlr.org/papers/v3/reunanen03a.html
Ross, B. C. (2014). Mutual information between discrete and continuous data sets. PLoS ONE, 9(2), e87357. https://doi.org/10.1371/journal.pone.0087357
Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B, 58(1), 267-288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
Whitney, A. W. (1971). A direct method of nonparametric measurement selection. IEEE Transactions on Computers, C-20(9), 1100-1103. https://doi.org/10.1109/T-C.1971.223410
Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
Xue, B., Zhang, M., Browne, W. N. y Yao, X. (2016). A survey on evolutionary computation approaches to feature selection. IEEE Transactions on Evolutionary Computation, 20(4), 606-626. https://doi.org/10.1109/TEVC.2015.2504420
# Referencias de Persona A
<!-- Kennedy y Eberhart (1997), Kirkpatrick et al. (1983), Glover (1986), Deb et al. (2002), pymoo... -->
