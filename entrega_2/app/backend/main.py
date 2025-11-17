from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn


# Definir estructura de input


class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# Cargar el modelo

# Crear la app

app = FastAPI(
    title="Predicción de compra",
    description="API para predecir si comportamiento de compra de bebidas",
    version="1.0",
)


# Ruta GET / home


@app.get("/")
def home():
    # Retornar información básica sobre la API
    # Formato de input y output
    return {
        "message": "Modelo de clasificación de potabilidad de agua",
        "input": {
            "ph": "float",
            "Hardness": "float",
            "Solids": "float",
            "Chloramines": "float",
            "Sulfate": "float",
            "Conductivity": "float",
            "Organic_carbon": "float",
            "Trihalomethanes": "float",
            "Turbidity": "float",
        },
        "output": {"Potable": "0 = No potable, 1 = Potable"},
    }


# Ruta POST /potabilidad/


@app.post("/prediction/")
def trigger_dag_run():
    endpoint = f"{AIRFLOW_URL}/dags/{DAG_ID}/dagRuns"
    data = {"conf": {"key": "value"}}  # Optional DAG run config
    response = requests.post(endpoint, json=data, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200 or response.status_code == 201:
        return f"DAG {DAG_ID} triggered successfully!"
    else:
        return f"Error triggering DAG: {response.text}"


# Ejecutar con uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
