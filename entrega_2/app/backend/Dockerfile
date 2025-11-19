# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install  -r requirements.txt

COPY  main.py .
COPY  dags .
COPY  models .

RUN airflow db migrate

EXPOSE 8000
RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin
 
VOLUME ["/base_data"]

CMD ["airflow", "standalone"]
CMD ["python", "main.py"]