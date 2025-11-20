FROM python:3.10-slim

WORKDIR /app

# Copia los archivos de backend
COPY app/backend/ .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia archivos necesarios para el modelo
COPY airflow/dags/model_functions.py /app/
COPY airflow/dags/constants.py /app/
# Expone el puerto para FastAPI
EXPOSE 8000

# Arranca FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]