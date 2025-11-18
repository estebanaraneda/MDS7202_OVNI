# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY  main.py .
COPY  dags .

EXPOSE 8000

VOLUME ["/data"]

CMD ["python", "main.py"]