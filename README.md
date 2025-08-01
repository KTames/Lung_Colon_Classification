# Clasificación del dataset LC25000 utilizando ResNet50, InceptionV4 y YOLOv11

Creado por Kendall Tames y Kevin Trejos para la materia de Visión Computacional de la Universidad de Costa Rica.

El dataset original se puede encontrar [aquí](https://www.kaggle.com/datasets/javaidahmadwani/lc25000).

El archivo `0. Descargar dataset.sh` descarga y limpia el dataset, lo deja con el nombre correcto, listo para utilizarse. No hace falta ejecutarlo puesto que la carpeta de `lung_colon_image_set` está disponible en el repositorio.

`1. Entrenamiento de modelos.ipynb` contiene el código para instanciar y entrenar los tres modelos.

`2. Métricas de modelos.ipynb` toma el mejor resultado surgido del entrenamiento para cada modelo, lo carga y obtiene métricas que luego son analizadas en el artículo para el proyecto primario.

`shared_definitions.py` es el núcleo del trabajo, ya que define modelos y funcionalidades compartidas que simplifican el código dentro de los cuadernos.
