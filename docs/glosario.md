# Glosario: el proyecto explicado desde cero

Todo el proyecto en una frase: tenemos una tabla con muchas columnas y queremos quedarnos con las pocas que de verdad sirven para predecir; como no se pueden probar todas las combinaciones, usamos estrategias de búsqueda inteligentes, las comparamos con métodos sencillos de forma justa y comprobamos con datos guardados que lo encontrado funciona de verdad.

## 1. Característica o variable

Imagina que quieres predecir si a un paciente le dará diabetes. Tienes una tabla con edad, peso, tensión, glucosa, horas de sueño, color de ojos, número de zapato… Cada columna es una **característica** o **variable**. Lo que quieres adivinar (diabetes sí/no) es la **etiqueta**.

## 2. Modelo o clasificador

Un programa que aprende de ejemplos: le das pacientes con su respuesta conocida, encuentra patrones y luego predice para pacientes nuevos. **Precisión** = porcentaje de aciertos; **error** = porcentaje de fallos.

**k-NN (vecinos más cercanos):** para un paciente nuevo busca los 5 más parecidos que ya conoce y copia lo que les pasó a la mayoría. Lo usamos porque es muy rápido y lo vamos a entrenar miles de veces.

## 3. Por qué elegir variables

Meter todas no siempre es mejor: el número de zapato no ayuda y además confunde al modelo (es ruido). Con menos variables el modelo es más rápido, más barato y más fácil de explicar.

La dificultad: cada variable es un interruptor encendido/apagado. Con 3 variables hay 2×2×2 = 8 combinaciones; con 500 hay 2^500, un número de unas 150 cifras. Imposible probarlas todas: eso significa "2^n soluciones posibles".

Una solución se escribe como una fila de ceros y unos, por ejemplo `1 0 0 1 1 0…` (1 = uso la variable, 0 = no). Es un **vector binario**.

## 4. Función objetivo y enfoque wrapper

La **función objetivo** es la nota que le ponemos a cada solución; buscamos la de mejor nota.

En el enfoque **wrapper** ("envoltorio") la nota se calcula entrenando el modelo solo con esas variables y midiendo cuánto se equivoca. Es como probarse ropa: fiable, pero lento. Nuestra nota mezcla equivocarse poco (lo principal) y usar pocas variables (desempata).

## 5. Metaheurística

Una **heurística** es un truco razonable para encontrar una buena solución sin probarlo todo. Una **metaheurística** es una estrategia general de búsqueda que sirve para muchos problemas. No garantiza la mejor solución, pero sí una muy buena en tiempo razonable.

**Algoritmo genético (GA)** — la evolución:

- Empiezas con una "población" de, por ejemplo, 50 soluciones al azar.
- Las mejores se combinan de dos en dos para crear hijos (**cruce**).
- De vez en cuando cambias un bit al azar (**mutación**) para que aparezcan cosas nuevas.
- Generación tras generación, la población mejora.

**PSO binario (enjambre de partículas)** — una bandada buscando comida:

- Cada "pájaro" es una solución que se mueve por el espacio de búsqueda.
- Se deja influir por el mejor sitio que él ha visto y por el mejor que ha visto la bandada.
- "Binario" = adaptado a soluciones de ceros y unos.

**Recocido simulado (SA)** — el metal que se enfría:

- Partes de una solución y pruebas un pequeño cambio (encender o apagar una variable).
- Si mejora, lo aceptas; si empeora, a veces también, sobre todo al principio.
- Esa probabilidad baja con el tiempo (la "temperatura" se enfría).
- Aceptar cosas peores evita quedarse atascado: bajar un poco de una colina para poder subir a una montaña más alta.

**Búsqueda tabú** — búsqueda con memoria:

- En cada paso haces el mejor cambio posible.
- Apuntas los cambios recientes en una lista negra (**lista tabú**) y los prohíbes unos pasos.
- Así no das vueltas en círculo.

## 6. Métodos clásicos (para comparar)

Sirven para responder: ¿las metaheurísticas aportan algo, o un método sencillo lo hace igual?

- **Selección secuencial hacia delante (SFS):** empiezas sin variables y añades una a una la que más mejora el modelo, como formar un equipo eligiendo al mejor jugador disponible cada vez. Rápido, pero puede perderse combinaciones que solo funcionan juntas.
- **Filtro por información mutua:** puntuar cada variable según cuánto "dice" de la respuesta, mirándolas de una en una y sin entrenar modelo; te quedas con las mejores. Muy rápido, pero no ve combinaciones.
- **LASSO:** un modelo al que se castiga por usar variables mientras aprende, así que pone a cero las que no aportan. Las que sobreviven son las elegidas.
- **Búsqueda aleatoria:** probar grupos al azar. La referencia mínima: si un método no le gana a esto, no sirve.

## 7. Biobjetivo, frente de Pareto y NSGA-II

Queremos dos cosas que chocan: acertar mucho y usar pocas variables. En vez de una sola respuesta, damos el menú de mejores compromisos, por ejemplo: 5 variables → 85 % de acierto; 10 → 89 %; 20 → 90 %. Ninguna opción es peor que otra en las dos cosas a la vez: eso es el **frente de Pareto** (como elegir móvil: más batería o más ligero).

**NSGA-II** es un algoritmo genético adaptado para encontrar ese menú entero de una vez. El **hipervolumen** es un número que mide lo bueno que es el menú: cuanto mayor, mejor.

## 8. Sobreajuste, validación cruzada y test

**Sobreajuste:** como el estudiante que memoriza el examen de prueba, saca un 10 ahí y suspende el real. Si buscas mucho, puedes encontrar variables que funcionan genial con tus datos por pura casualidad.

- **Partición de test:** antes de empezar guardas el 30 % de los datos en un cajón cerrado que la búsqueda nunca ve. Al final es el examen real.
- **Validación cruzada (CV):** con el 70 % restante, para evaluar cada solución divides en 5 trozos, entrenas con 4 y pruebas con el quinto, 5 veces rotando, y haces la media. Cinco exámenes de prueba en vez de uno: nota más fiable.

Si la nota de CV es mucho mejor que la de test, hubo sobreajuste, y eso también es un resultado a comentar.

## 9. Términos de la experimentación

- **Semilla:** fija el azar para poder repetir exactamente un resultado. Ejecutamos cada método con 10–30 semillas para no fiarnos de un día de suerte.
- **Media y desviación típica:** la media es el resultado típico; la desviación, cuánto varía entre ejecuciones (¿es fiable o una lotería?).
- **Presupuesto de evaluaciones:** todos los métodos tienen el mismo número de pruebas (p. ej. 3.000 modelos entrenados). Carrera justa.
- **Tests estadísticos (Friedman, Wilcoxon):** dicen si una diferencia ("A es un 1 % mejor que B") es real o casualidad. Friedman compara todos a la vez; Wilcoxon, de dos en dos.
- **Curva de convergencia:** gráfico de cómo mejora la mejor solución con las pruebas; muestra quién mejora rápido y quién se estanca.
- **Caché:** apuntar las notas ya calculadas para no repetir el mismo entrenamiento.

## 10. Los datos

- **Madelon:** datos artificiales de una competición de 2003 sobre este mismo problema. 500 variables, de las que solo unas 20 sirven; el resto es ruido puesto a propósito. Perfecto para comprobar si los métodos encuentran las buenas.
- **Arcene y microarrays:** datos médicos (actividad de miles de genes) con unas 10.000 variables y muy pocos pacientes. Más difíciles y lentos: opcionales.

## 11. Modelo sustituto ("para ir más allá")

Entrenar el modelo real para cada solución es lo lento. Un **modelo sustituto** es un "adivinador" barato que predice la nota de una solución sin entrenarla. Descarta las candidatas que seguramente son malas y solo las prometedoras se evalúan de verdad, como un primer filtro de currículums antes de las entrevistas.
