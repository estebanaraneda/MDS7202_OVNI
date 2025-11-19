from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
import requests


BACKEND_URL = "http://backend:8000/prediction"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "admin"
# Definir estructura de input

# Crear la app

app = FastAPI(
    title="Predicción de compra",
    description="API para predecir el comportamiento de compra de bebidas",
    version="1.0",
)


# Ruta GET / home
def trigger_dag_run(data: InputData):
    dag_id="model_predictor"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": f"manual__{dag_id}",
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
# ---------------------------------------------------------
# Helper function: Trigger DAG via Airflow REST API
# ---------------------------------------------------------
def trigger_dag_run(payload: dict):
    dag_id = "model_predictor"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"

    response = requests.post(
        url,
        json=payload,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
    )
    response.raise_for_status()
    return response.json()  # contains dag_run_id


# ---------------------------------------------------------
# Helper function: Poll DAG status until finished
# ---------------------------------------------------------
def wait_for_dag_run(dag_run_id: str, timeout=60):
    dag_id = "model_predictor"
    status_url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"

    start = time.time()

    while True:
        response = requests.get(status_url, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
        response.raise_for_status()
        state = response.json()["state"]

        if state in ["success", "failed"]:
            return state

        if time.time() - start > timeout:
            return "timeout"

        time.sleep(2)


# ---------------------------------------------------------
# Helper: Fetch XCom from DAG result
# ---------------------------------------------------------
def get_dag_xcom(dag_run_id: str):
    dag_id = "model_predictor"
    url = (
        f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/"
        f"predict_task/xcomEntries"
    )

    response = requests.get(url, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
    response.raise_for_status()
    xcom_list = response.json()["xcom_entries"]

    # Expecting XCom key: prediction
    for entry in xcom_list:
        if entry["key"] == "prediction":
            return entry["value"]

    return None
@app.get("/")
def home():
    # Retornar información básica sobre la API
    # Formato de input y output
    return {
        "message": "Modelo de predicción de compra bebestibles",
        "input": {"algo": "float"}, "output": {"Preddiccion"}}


# Ruta POST /potabilidad/


@app.post("/prediction/")
    # 1) Trigger DAG run
    payload = {
        "dag_run_id": f"manual__{int(time.time())}",
        "conf": sample.model_dump()
    }

    dag_run = trigger_dag_run(payload)
    dag_run_id = dag_run["dag_run_id"]

    # 2) Wait for DAG to complete
    state = wait_for_dag_run(dag_run_id)

    if state != "success":
        return {"error": f"DAG finished with state: {state}"}

    # 3) Get prediction from XCom
    prediction = get_dag_xcom(dag_run_id)

    return {"prediction": prediction}


# Ejecutar con uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
