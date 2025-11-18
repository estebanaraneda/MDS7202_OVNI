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
    dag_id="model_predictor"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": f"manual__{dag_id}__{{{{ ts_nodash }}}}",
        "conf": {"source": "gradio-ui"},   # let Airflow use execution_date = now
        "note": "Triggered from Gradio UI"}
    try:
        response = requests.post(
            url,
            json=payload,
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
        )
        return f"Status {response.status_code}: {response.text}"
    except Exception as e:
        return str(e)


# Ejecutar con uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
