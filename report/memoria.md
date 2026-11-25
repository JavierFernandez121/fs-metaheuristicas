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
Tabu	0,0106 (0,0002)	tenure = 30	0,0129 (0,0007)	peor; se mantiene el valor automatico
NSGA-II	0,0112 (0,0004)	todas	0,0111 a 0,0116	ruido; se mantiene
En tabu, todas las configuraciones alcanzaron el mismo error de validacion cruzada (0,0075); solo cambia, a lo sumo, en una variable.
Limitaciones. (1) Breast Cancer es un problema facil y muchas configuraciones saturan en el mismo error, por lo que discrimina poco: conviene comprobar los dos cambios adoptados en Madelon. (2) Variar un parametro cada vez no detecta interacciones. (3) BPSO se probo con solo 3 semillas, asi que su error estandar es poco fiable. (4) Con unas 40 configuraciones probadas, alguna mejora limite puede deberse al azar, por eso solo se adoptaron las claramente por encima del ruido.

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
# 5.11 Metodos clasicos · B
<!-- Lo escribe Persona B -->
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
