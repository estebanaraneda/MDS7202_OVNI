# Conclusiones de proyecto hito 2, grupo OVNI
En este hito del proyecto se implementó un pipeline que resume 
lo realizado en el hito 1 e integra el procesamiento en un pipeline de Airflow,
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
En este hito so ocupo diferentes herramientas vistas en el curso.
Esto va más hallá de crear modelos como en el hito 1. Como grupo 
tuvimos que aplicar difrenetes habilidades que un cientifico de datos ha
de poseer especificamente las habilidades de crear un entregable autonomo
y funcional, el cual pueda ser entregado y ocupado por un usuario
con la ayuda de documentación proveída. 

### Airflow

### Aplicación
El despliegue en aplicación, aunque se busca que los contenedores actuen de
forma aislada la realidad es que ambos lados de la aplicación estan conectados
no solo entre sí, también con el usuario y el almacenamiento de los datos.
Un importante desafio a considerar es que cada llamada entre contenedores
entregue y reciba su parte de la consulta con los parámetros, argumentos,
llaves a API correcta y el formato correcto (como es html).

El realizar el frontend da libertad en el cómo presentar la aplicación
al usuario, por lo tanto, es nuestro deber organizar la interfaz
para que sea intuitiva y exlicativa, explicando el funcionamiento de la
aplicación y las intrucciones de uso.



### Trabajo futuro

### Reflexiones finales
La importancia de este hito del proyecto es ser la base para la entrega final
y el poner en practica las diferentes técnicas y conocimientos de MLOps
aprendidos durante el curso.





