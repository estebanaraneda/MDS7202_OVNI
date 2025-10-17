from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn


# Definir estructura de input


class WaterSample(BaseModel):
    # Reemplaza estos campos con los nombres de tus columnas
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


with open("models/best_model_pipeline_20251017_181200.pkl", "rb") as f:  # Reemplaza con el nombre correcto del archivo
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
    input_df = pd.DataFrame([sample.dict()])

    # Hacer predicción
    pred = model.predict(input_df)[0]  # 0 o 1
    # prob = model.predict_proba(input_df)[0][1]  # probabilidad de ser potable

    return {"prediction": int(pred)}


# Ejecutar con uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
