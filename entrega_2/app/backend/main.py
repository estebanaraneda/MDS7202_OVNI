from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Backend de Predicciones", description="API que carga modelos generados por Airflow", version="1.0.0"
)

MODEL_DIR = "/models"


def get_latest_model_path():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_model_pipeline_") and f.endswith(".pkl")]
    print(f"🔹 Model files found: {model_files}")
    if not model_files:
        return None
    return os.path.join(MODEL_DIR, sorted(model_files)[-1])


def load_model():
    model_path = get_latest_model_path()
    print(f"🔹 Path del modelo usado: {model_path}")
    if model_path is None:
        print("⚠ No se encontró ningún modelo en /models")
        return None

    try:
        print(f"♻️  Recargando modelo desde: {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return None


class InputData(BaseModel):
    purchase_date: str
    customer_type: str
    num_deliver_per_week: str
    brand: str
    segment: str
    package: str
    Y: float
    X: float
    size: float
    items_lag_1: float
    items_lag_2: float
    items_lag_3: float
    items_lag_4: float
    num_orders_lag_1: float
    num_orders_lag_2: float
    num_orders_lag_3: float
    num_orders_lag_4: float


@app.get("/")
def home():
    return {"status": "Backend funcionando"}


@app.post("/predict")
def predict(data: InputData):

    # 🔥 Recarga del modelo EN CADA REQUEST
    model = load_model()

    if model is None:
        return {"error": "Modelo no encontrado en /models"}

    df = pd.DataFrame([data.dict()])
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    prediction = model.predict(df)[0]
    print(f"✅ Predicción realizada: {prediction}")
    # Convertir numpy → python
    prediction = prediction.item()
    print(f"✅ Predicción realizada: {prediction}")
    return {"input": data.dict(), "prediction": prediction}
