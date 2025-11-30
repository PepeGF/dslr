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