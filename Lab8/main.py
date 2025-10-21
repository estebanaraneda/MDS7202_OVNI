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


with open("models/best_model_pipeline_20251020_213211.pkl", "rb") as f:
    model = pickle.load(f)


# Crear la app

app = FastAPI(
    title="Water Potability API",
    description="API para predecir si una muestra de agua es potable usando XGBoost optimizado con Optuna",
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


@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    # Convertir el input a DataFrame
    input_df = pd.DataFrame([sample.model_dump()])

    # Hacer predicción
    pred = model.predict(input_df)[0]  # 0 o 1
    # prob = model.predict_proba(input_df)[0][1]  # probabilidad de ser potable

    return {"prediction": int(pred)}


# Ejecutar con uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
