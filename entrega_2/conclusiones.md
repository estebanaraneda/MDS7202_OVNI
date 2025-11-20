# Conclusiones de proyecto hito 2, grupo OVNI

En este hito del proyecto se implementó un pipeline que resume lo realizado en el hito 1 e integra el procesamiento en un pipeline de Airflow,
incluido los pasos de:

- Extracción de datos.
- Limpieza y transformación
- Reentrenamiento del modelo
- Generación de predicciones

Posteriormente se realiza una aplicación web ocupando la herramienta Docker
organizada de la sigiente forma:

- Backend con FastAPI
- Frontend con Gradio

### Enfoque

En este hito so ocupo diferentes herramientas vistas en el curso. Esto va más allá de crear modelos como en el hito 1. Como grupo
tuvimos que aplicar diferentes habilidades que un cientifico de datos ha de poseer especificamente las habilidades de crear un entregable autonomo
y funcional, el cual pueda ser entregado y ocupado por un usuario con la ayuda de documentación proveída.

## Tracking

El uso de **MLflow** como herramienta de *tracking* mejoró significativamente la organización y eficiencia del proceso de desarrollo.

Gracias a MLflow fue posible:

* Registrar automáticamente parámetros, métricas y artefactos del modelo.
* Comparar distintas ejecuciones de entrenamiento de manera clara.
* Versionar los modelos y garantizar reproducibilidad.
* 

## Desafíos del despliegue con FastAPI y Gradio

El despliegue con **FastAPI** y **Gradio** presentó varios desafíos técnicos, entre ellos:

* Manejo correcto de la serialización del modelo y sus dependencias.
* Coordinación entre frontend, backend y contenedores Docker.
* Validación del formato de entrada y estandarización del JSON enviado desde el frontend.
* Problemas comunes de rutas, paquetes e importaciones al mover el modelo a un entorno aislado.

## Aportes de Airflow a la robustez y escalabilidad del pipeline

La incorporación de **Airflow** mejoró significativamente la robustez del pipeline:

* Permitió automatizar el entrenamiento del modelo con una estructura clara.
* Proporcionó manejo de logs, monitoreo, visualización del DAG y reintentos automáticos.
* Aseguró un flujo reproducible donde nuevos modelos se generan de forma periódica.

### Aplicación

El despliegue en aplicación, aunque se busca que los contenedores actuen de forma aislada la realidad es que ambos lados de la aplicación estan conectados
no solo entre sí, también con el usuario y el almacenamiento de los datos. Un importante desafio a considerar es que cada llamada entre contenedores
entregue y reciba su parte de la consulta con los parámetros, argumentos.

El realizar el frontend da libertad en el cómo presentar la aplicación al usuario, por lo tanto, es nuestro deber organizar la interfazpara que sea intuitiva y explicativa.

### Trabajo futuro

- Implementar detección de drift para el reentreno del modelo, ya que actualmente solo se cuenta con reentreno periodico.
- Mejorar la forma de carga del modelo en el backend, ya que actualmente lo recarga cada vez que se hace una predicción, para tomar el más nuevo, en su lugar debiera recargar solo si hay nuevos modelos.

### Reflexiones finales

La importancia de este hito del proyecto es ser la base para la entrega final y el poner en practica las diferentes técnicas y conocimientos de MLOps aprendidos durante el curso.
