import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr


def create_folders(**kwargs):
    """
    Crea una carpeta con la fecha de ejecución (proveniente del DAG)
    y tres subcarpetas: 'raw', 'splits' y 'models'.

    Parameters:
        **kwargs: Recibe los parámetros del contexto de Airflow.
                  Usa 'ds' (date string, formato YYYY-MM-DD).
    """
    # Obtener la fecha de ejecución desde Airflow (YYYY-MM-DD)
    execution_date = kwargs.get("ds")

    # Crear carpeta principal
    base_folder = execution_date
    os.makedirs(base_folder, exist_ok=True)

    # Crear subcarpetas
    subfolders = ["raw", "splits", "models"]
    for sub in subfolders:
        os.makedirs(os.path.join(base_folder, sub), exist_ok=True)

    print(f"Carpetas creadas en: {os.path.abspath(base_folder)}")


def split_data(**kwargs):
    """
    Lee 'data_1.csv' desde la carpeta 'raw', aplica un hold-out split (80/20),
    manteniendo la proporción de la variable objetivo, y guarda los resultados
    en la carpeta 'splits'.

    Parameters:
        **kwargs: (opcional) Permite recibir argumentos del DAG de Airflow.
                  Se usa 'ds' si se quiere acceder a la fecha de ejecución.
    """
    # Obtener fecha desde Airflow (si se pasa como contexto)
    execution_date = kwargs.get("ds")

    # Definir rutas de entrada y salida
    base_folder = execution_date
    raw_path = os.path.join(base_folder, "raw", "data_1.csv")
    splits_folder = os.path.join(base_folder, "splits")
    os.makedirs(splits_folder, exist_ok=True)

    # Leer el dataset original
    df = pd.read_csv(raw_path)

    # Definir variable objetivo
    target_col = "HiringDecision"

    # Dividir los datos (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

    # Guardar los splits
    train_df.to_csv(os.path.join(splits_folder, "train.csv"), index=False)
    test_df.to_csv(os.path.join(splits_folder, "test.csv"), index=False)

    print(f"Conjuntos guardados en: {os.path.abspath(splits_folder)}")


def preprocess_and_train(**kwargs):
    """
    Lee los datasets de entrenamiento y prueba desde la carpeta 'splits',
    aplica un pipeline con preprocesamiento (ColumnTransformer) y entrena
    un modelo RandomForest.
    Guarda el pipeline entrenado como archivo joblib en 'models'
    e imprime accuracy y F1-score de la clase positiva ('contratado').

    Parameters:
        **kwargs: permite obtener la fecha de ejecución desde Airflow (ds)
    """
    # Obtener la fecha desde Airflow o usar la actual
    execution_date = kwargs.get("ds")

    # Definir rutas
    base_folder = execution_date
    splits_folder = os.path.join(base_folder, "splits")
    models_folder = os.path.join(base_folder, "models")
    os.makedirs(models_folder, exist_ok=True)

    # Leer datasets
    train_df = pd.read_csv(os.path.join(splits_folder, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_folder, "test.csv"))

    # Definir variable objetivo
    target_col = "HiringDecision"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Identificar columnas numéricas y categóricas
    categorical_features = ["Gender", "RecruitmentStrategy"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    # Definir transformadores
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Crear pipeline con preprocesamiento + modelo
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Entrenar
    clf.fit(X_train, y_train)

    # Evaluar
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)  # clase positiva

    print(f"Accuracy en test: {acc:.3f}")
    print(f"F1-score (clase 'contratado'): {f1:.3f}")

    # Guardar modelo entrenado
    model_path = os.path.join(models_folder, "trained_pipeline.joblib")
    joblib.dump(clf, model_path)

    print(f"Modelo guardado en: {os.path.abspath(model_path)}")


def predict(file, model_path):
    """
    Carga el modelo desde 'model_path', lee un archivo JSON y realiza predicciones.
    Retorna un diccionario con la etiqueta predicha.
    """
    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)

    print(f"La predicción es: {predictions}")

    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]
    return {"Predicción": labels[0]}


def gradio_interface(**kwargs):
    """
    Despliega una interfaz Gradio para hacer predicciones con el modelo entrenado.
    Carga el modelo desde la carpeta 'models' según la fecha de ejecución.
    Compatible con Airflow (usa kwargs['ds'] si está disponible).
    """
    # Obtener fecha desde Airflow
    execution_date = kwargs.get("ds")

    # Construir la ruta del modelo entrenado
    model_path = os.path.join(execution_date, "models", "trained_pipeline.joblib")

    # Verificar existencia del modelo
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    # Crear interfaz de Gradio
    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no.",
    )

    # Lanzar interfaz
    interface.launch(share=True)
