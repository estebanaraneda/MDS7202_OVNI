import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib


def create_folders(**kwargs):
    """
    Crea una carpeta principal usando la fecha de ejecución y las subcarpetas:
    raw, preprocessed, splits, models
    """
    execution_date = kwargs.get("ds")
    base_folder = execution_date
    os.makedirs(base_folder, exist_ok=True)

    subfolders = ["raw", "preprocessed", "splits", "models"]
    for sub in subfolders:
        os.makedirs(os.path.join(base_folder, sub), exist_ok=True)

    print(f"Carpetas creadas en: {os.path.abspath(base_folder)}")


def load_ands_merge(**kwargs):
    """
    Lee data_1.csv y data_2.csv desde raw, los concatena y guarda el resultado en preprocessed
    """
    execution_date = kwargs.get("ds")
    base_folder = execution_date
    raw_folder = os.path.join(base_folder, "raw")
    preprocessed_folder = os.path.join(base_folder, "preprocessed")
    os.makedirs(preprocessed_folder, exist_ok=True)

    files = ["data_1.csv", "data_2.csv"]
    dfs = []

    for file in files:
        file_path = os.path.join(raw_folder, file)
        if os.path.exists(file_path):
            dfs.append(pd.read_csv(file_path))

    if not dfs:
        raise FileNotFoundError("No se encontraron archivos data_1.csv ni data_2.csv en raw")

    merged_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(preprocessed_folder, "merged_data.csv")
    merged_df.to_csv(output_path, index=False)

    print(f"Datos concatenados y guardados en: {output_path}")


def split_data(**kwargs):
    """
    Realiza hold-out split (80/20) del dataset preprocesado y guarda train/test en splits
    """
    execution_date = kwargs.get("ds")
    base_folder = execution_date
    preprocessed_folder = os.path.join(base_folder, "preprocessed")
    splits_folder = os.path.join(base_folder, "splits")
    os.makedirs(splits_folder, exist_ok=True)

    data_path = os.path.join(preprocessed_folder, "merged_data.csv")
    df = pd.read_csv(data_path)

    target_col = "HiringDecision"
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])

    train_df.to_csv(os.path.join(splits_folder, "train.csv"), index=False)
    test_df.to_csv(os.path.join(splits_folder, "test.csv"), index=False)

    print(f"Train/test guardados en: {os.path.abspath(splits_folder)}")


def train_model(model, **kwargs):
    """
    Entrena un modelo de clasificación usando un Pipeline con preprocesamiento
    y guarda el pipeline entrenado en models
    """
    execution_date = kwargs.get("ds")
    base_folder = execution_date
    splits_folder = os.path.join(base_folder, "splits")
    models_folder = os.path.join(base_folder, "models")
    os.makedirs(models_folder, exist_ok=True)

    train_df = pd.read_csv(os.path.join(splits_folder, "train.csv"))
    target_col = "HiringDecision"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    categorical_features = ["Gender", "RecruitmentStrategy"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    pipeline.fit(X_train, y_train)

    model_name = f"{model.__class__.__name__}_pipeline.joblib"
    joblib.dump(pipeline, os.path.join(models_folder, model_name))

    print(f"Modelo entrenado y guardado como: {model_name}")


def evaluate_models(**kwargs):
    """
    Evalúa todos los modelos en models usando accuracy sobre test y guarda el mejor modelo
    """
    execution_date = kwargs.get("ds")
    base_folder = execution_date
    splits_folder = os.path.join(base_folder, "splits")
    models_folder = os.path.join(base_folder, "models")

    test_df = pd.read_csv(os.path.join(splits_folder, "test.csv"))
    target_col = "HiringDecision"
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    best_acc = -1
    best_model_name = None

    for file in os.listdir(models_folder):
        if file.endswith(".joblib"):
            model_path = os.path.join(models_folder, file)
            pipeline = joblib.load(model_path)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{file}: Accuracy = {acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                best_model_name = file

    if best_model_name:
        best_model_path = os.path.join(models_folder, "best_model.joblib")
        joblib.dump(joblib.load(os.path.join(models_folder, best_model_name)), best_model_path)
        print(f"Mejor modelo seleccionado: {best_model_name} con Accuracy = {best_acc:.3f}")
        print(f"Guardado como: {best_model_path}")
    else:
        print("No se encontraron modelos para evaluar.")
