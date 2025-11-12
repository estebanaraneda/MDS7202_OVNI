# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /root/airflow

# Establece la variable de entorno AIRFLOW_HOME
ENV AIRFLOW_HOME=/root/airflow

# Configura Airflow para no cargar ejemplos
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN python -m pip install --upgrade pip

# Instala Apache Airflow y otras librerías
COPY requirements.txt .
RUN pip install -r requirements.txt

# Inicializa la base de datos de Airflow
RUN airflow db migrate

# Expone el puerto 8080 para el servidor web de Airflow
EXPOSE 8080

# Crea el usuario admin de Airflow
RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

# Copia las carpetas necesarias al contenedor
COPY ./airflow/dags $AIRFLOW_HOME/dags
COPY ./airflow/logs $AIRFLOW_HOME/logs
COPY ./airflow/plugins $AIRFLOW_HOME/plugins
COPY ./airflow/base_data $AIRFLOW_HOME/base_data

# Comando para iniciar el servidor web y el scheduler
CMD ["airflow", "standalone"]
