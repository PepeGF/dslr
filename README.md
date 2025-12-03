
# DSLR

  

Documentación del proyecto DSLR, basado en **regresión logística**.

Se divide en 4 partes:

*- Análisis de datos.*

*- Visualización de datos.*

*- Regresión logística.*

*- Bonus.*

  

El objetivo de este proyecto es sustituir al *Sombrero Seleccionador* de la Escuela de Magia Hogwarts, sí como suena...

Se parte de un **dataset de entrenamiento** con datos de 1600 estudiantes de la escuela entre los que se incluyen sus nombres, la casa a la que pertenecen, apellidos, fechas de nacimiento, sus manos buenas y las notas en 13 asignaturas.

Con estos datos se debe asignar la casa correspondiente a otros 400 estudiantes que por alguna razón no tienen casa asignada.

Para dar el proyecto como válido se debe de obtener una **precisión de más del 98%** en la predicción.

  

El lenguaje a utilizar es libre, se ha hecho con **Python 3.12** y el gestor **UV**, procurando seguir las reglas de estilo de **PEP**.

  

Los archivos con los datasets de entrenamiento y de test deben de estar en una subcarpeta *data*.

  
  

## Análisis de datos

Como primer paso para familiarizarse con el conjunto de datos hay que escribir un programa que emule el comportamiento del método *describe* de pandas, sin utilizar funciones de librerías que hagan los cálculos. Todos los cálculos, desde min/max a medias y desviaciones estándar se realizan con funciones propias.

Para ejecutarlo: 

    uv run describe.py dataset_train.csv

o bien 

    python describe.py dataset_train.csv
 

## Visualización de datos

En esta segunda parte se trata de representar gráficamente la información para tener una comprensión mejor de cómo son los datos.

### Histograma
Con el histograma se intenta encontrar qué asignatura tiene una distribución más uniforme para las cuatro casas.
Se utiliza la librería **matplotlib** para representar el histograma.
Se ejecuta con el comando

    uv run histogram.py dataset_train.csv

o

    python histogram.py dataset_train.py

Se muestran todos los histogramas en bucle rápidamente y al terminar la representación individual se muestra un collage de todos los histogramas para poder compararlos.

### Scatter plot
Con este gráfico se muestra qué dos asignaturas tienen distribuciones similares.
Como se puede interpretar de varias formas se representan 3 opciones diferentes.
- En la primera se ve la estrechísima relación, casi perfecta, entre las asignaturas.
- En la segunda se muestran dos asignaturas que para todas las casas tienen distribuciones similares que no guardan correlación.
- En la última la distribución es similar para las dos asignaturas, aparentemente aleatoria para las cuatro casas.

Se ejecuta con el comando

    uv run scatter_plot.py

o bien

    python scatter_plot.py

Se usa un estilo predefinido para simplificar el código lo máximo posible sin perder funcionalidad.

### Pair plot
En esta sección se muestran todos los gráficos de pares para todas las combinaciones de asignaturas.
Cuando en la matrix coincide la asignatura en ambos ejes se representa un histograma de la asignatura correspondiente.
En este caso se utiliza la librería **seaborn**, que genera gráficos con aún menos código, y un cuaderno de **Jupyter Notebook** en el que se muestra el gráfico y se guarda como archivo de imagen png.
Con este gráfico se decide más fácilmente qué asignmaturas son más útiles para el entrenamiento del modelo.

## Regresión Logística.
La regresión logística es una técnica estadística para predecir la probabilidad de un resultado binario (0 o 1) basándose en una o más variables predictoras.
En este caso, al ser 4 las posibilidades de clasificación se utiliza la estrategia "one vs all" en la que se clasifica si un estudiante pertenece a una determinada casa o no. Este proceso se repite para todas las casas y el resultado que genera una mayor probabilidad de pertenencia a una casa se elige como casa definitiva para el estudiante.

La clasificación se basa en la ecuación **Sigmoide**, en la que la probabilidad $\sigma$ de que un elemento sea clasificable como 1 es:

$$
\sigma(z) = \frac{1}{1 + e^{z}}
$$

Esta función es monótonamente creciente desde -$\infin$ hasta $\infin$, con una asíntota horizontal de valor 0 en -$\infin$ y otra asíntota horizontal del valor 1 en $\infin$, teniendo su punto medio en $z = 0$, donde $\sigma = 1/2$.

![Representación gráfica sigmoide](doc/sigmoide.png)

donde $z = w * x + b$, 
siendo $w$ la matriz de los pesos y $b$ el vector de los *bias* (prejuicios iniciales).

### Entrenamiento del modelo
Para entrenar el modelo, es decir, calcular *w* y *b* que después se usarán para la predicción, se utiliza la técnica de *Descenso de gradiente*, en la que el objetivo es *minimizar la función de coste*.

La **función de coste** se define como:

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}y^i · log(h_\theta(x^i)) + (1-y^i)·log(1-h_\theta(x^i))
$$

donde 

$$h_\theta = g(\theta^T·x)$$ 

siendo 

$$g(z) = \frac{1}{1 + e^{-z}}$$

En cada iteración del entrenamiento se calcula $w$ y $b$:

$$
w = w - \alpha·\frac{\partial{J}(w, b)}{\partial{w}}
$$

y

$$
b = b - \alpha·\frac{\partial{J}(w, b)}{\partial{b}}
$$

donde $\alpha$ es el ratio de aprendizaje (learning rate), un factor de corrección que permite avanzar poco a poco en la dirección en la que la pendiente de la función de coste es máxima. Como se pretende reducir la función de coste se avanza en sentido contrario, por eso es necesario el signo "-".
Aplicando la regla de la cadena de las derivadas se llega a que la derivada parcial de la función de coste respecto a $w$ es:

 $$
 \frac{\partial{J}(w, b)}{\partial{w_j}} = \frac{1}{m}\sum_{i=1}^{m}(g_i-y_i)·x_{i,j}
 $$
 
 y la derivada parcial de la función de coste ($J$) respecto a $b$ es:
 
 $$
 \frac{\partial{J}(w, b)}{\partial{b}} = \frac{1}{m}\sum_{i=1}^{m}(g_i-y_i)
 $$
 
donde $i$ es cada uno de los registros usados en el entrenamiento u $j$ es cada una de las características (en este caso las asignaturas) y $y$ es la predicción real de cada registro (1 o 0 dependiendo de si es la casa adecuada o no) y $g$ es la predicción calculada para cada registro.

Con este modelo de entrenamiento se premia las predicciones acertadas y se penalizan las falladas, haciendo en en cada iteración cada valor de la matriz $w$ y del vector $b$ se acerque al punto óptimo.

## Predicción
Una vez se han calculado los valores de $w$ y $b$ se guardan en un archivo *csv* para que puedan ser utilizados en el programa de predicción.
El script de predicción parte de los registros de estudiantes, que tienen la misma estructura que los datos de entrenamiento, con la salvedad de que no tienen la categoría ***Hogwarts House***, ya que es precisamente la que hay que calcular.
De los datos de predicción se extrae un dataframe con las notas de asignaturas usadas en el entrenamiento, al que se le aplica la ecuación sigmoide:

$$g(z) = \frac{1}{1 + e^{-z}}$$ 

recordemos que  $z = w * x + b$, para cada una de las casas, lo que generará 4 probabilidad de pertenencia a cada una de las 4 casas para cada uno de los registros de predicción.
Tomando la máxima de las probabilidades correspondientes a cada registro se obtiene la predicción de la casa a la que pertenece el estudiante.
Estos valores se guardan en el archivo *houses.csv* para ser comprobados posteriormente y obtener un ratio de acierto del modelo.

## Bonus

### Información adicional en describe.py
El método describe() de pandas genera información suficiente para conocer de forma somera cómo se distribuyen los datos, sin embargo se ha añadido el rango intercuartil (la diferencia entre el percentil 75 y el 25). Este dato ayuda a conocer cómo de concetrados están los datos en torno a la mediana y da una idea de cuántos valores lejanos hay.

### Matriz de confusión y parámetros de calidad
El parámetro de porcentaje de acierto no es el único criterio que determina la calidad del modelo.
El script *confusion_matrix.py* muestra la precisión de la **predicción** para cada una de las casas y el parámetro **recall** y el parámtro **F1 Score** que aglutina varios parámetros en uno solo.
También muestra un gráfico con una matriz de confusión que enseña las predicciones reales vs predichas.

### Modelo de entrenamiento adicional
El modelo utilizado *1 vs all* calcula las probabilidades de pertenencia o no para cada registro para cada una de las casas y posteriormente se escoge la mayor de ellas.
Esto obliga a repetir el proceso de entrenamiento una vez por cada casa, lo que para ejemplos en los que las categorías de clasificación fuesen mayores o el número de registro fuese muy grande sería muy ineficiente.
Por ello se ha creado otro modelo de entrenamiento, **Softmax** que calcula directamente la probabilidad de pertenencia a cada cada, siendo la suma de las probabilidades igual a 1 en todos los registros. 

## Fuentes de consulta

Para entender e implementar el modelo de regresión logística con la estrategia *1 vs all* ha sido muy útil:
https://www.youtube.com/watch?v=3giTXZbyf1Q
y
https://www.youtube.com/watch?v=S6iuhdYsGC8

Para el modelo *Softmax*:
https://peterroelants.github.io/posts/cross-entropy-logistic/
https://peterroelants.github.io/posts/cross-entropy-softmax/

## Posibles mejoras
 - Modelo de regresión por lotes
