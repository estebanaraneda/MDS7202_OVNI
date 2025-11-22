import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from constants import CATEGORICAL_VARIABLES, NUMERICAL_VARIABLES, TEMPORAL_VARIABLES
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import optuna
from joblib import dump, load
import mlflow
import shap
import matplotlib.pyplot as plt


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(self.lower_quantile)
        self.upper_bounds_ = X.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X

    def set_output(self, *, transform=None):
        return self


def date_features(df):
    df = df.copy()
    df["month"] = df["purchase_date"].dt.month
    df["day"] = df["purchase_date"].dt.day
    df["week"] = df["purchase_date"].dt.isocalendar().week
    return df.drop(columns=["purchase_date"])


def split_reading_data(execution_date):  # Definir rutas
    base_folder = execution_date
    splits_folder = os.path.join(base_folder, "splits")
    # Leer datasets
    train_df = pd.read_parquet(os.path.join(splits_folder, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_folder, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_folder, "test.parquet"))

    # Definir variable objetivo y columnas ID
    target_col = "order"
    id_cols = ["customer_id", "product_id"]

    # Drop columnas ID y separar X e y
    X_train = train_df.drop(columns=[target_col] + id_cols)
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col] + id_cols)
    y_val = val_df[target_col]
    X_test = test_df.drop(columns=[target_col] + id_cols)
    y_test = test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


def objective(trial):

    # Definimos los hiperparámetros a optimizar
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_leaves = trial.suggest_int("max_leaves", 1, 100)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 1.0)

    # Hiperparámetros one-hot encoder
    min_frequency = trial.suggest_int("min_frequency", 1, 10)
    # Cálculo de pesos de clases
    classes, classes_counts = np.unique(y_train, return_counts=True)
    classes_weights = {cls: sum(classes_counts) / count for cls, count in zip(classes, classes_counts)}
    print("Clases y sus conteos:", dict(zip(classes, classes_counts)))
    print(classes_weights)

    classifier_obj = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaves=max_leaves,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=classes_weights[1] / classes_weights[0],
    )

    # Identificamos las columnas numéricas y categóricas
    numerical_columns = NUMERICAL_VARIABLES
    categorical_columns = CATEGORICAL_VARIABLES + TEMPORAL_VARIABLES

    # Pipeline de preprocesamiento
    numerical_pipeline = Pipeline(
        steps=[("outlier_clip", OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)), ("scaler", StandardScaler())]
    )

    cat_transformer = Pipeline(
        [
            (
                "category_enc",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore", min_frequency=min_frequency),
            ),  # Target Encoding
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", cat_transformer, categorical_columns),
        ]
    ).set_output(transform="pandas")

    # Pipeline de optimización
    date_transformer = FunctionTransformer(date_features)

    model_to_optimize = Pipeline(
        steps=[("date_transformer", date_transformer), ("preprocessor", preprocessor), ("model", classifier_obj)]
    )

    # Inicio de una nueva corrida en MLflow
    mlflow.start_run(nested=True, run_name=f"trial_{trial.number}")
    # Entrenamos el modelo
    model_to_optimize.fit(X_train, y_train)
    # Realizamos las predicciones
    y_opt_val_pred = model_to_optimize.predict(X_val)
    # Calculamos la métrica de evaluación
    recall = (y_opt_val_pred[y_val == 1] == 1).mean()
    # Guardamos el mejor pipeline entrenado
    trial.set_user_attr("best_pipeline", model_to_optimize)

    # Logueo de parámetros y métricas en MLflow
    mlflow.log_params(trial.params)
    mlflow.log_metric("valid_recall", recall)

    # Guardado del modelo en MLflow
    signature = mlflow.models.infer_signature(X_train, model_to_optimize.predict(X_train))
    mlflow.sklearn.log_model(model_to_optimize, "model", signature=signature, input_example=X_train.iloc[:5])

    # Finalización de la corrida en MLflow
    mlflow.end_run()

    return recall


def optimize_model(**kwargs):

    # Setamos X_train, y_train, X_val, y_val, X_test, y_test como variables globales
    # Esto para no tener que leerlas dentro de la función objective en cada iteración
    global X_train, y_train, X_val, y_val, X_test, y_test

    execution_date = kwargs.get("ds")
    X_train, y_train, X_val, y_val, X_test, y_test = split_reading_data(execution_date)

    # print de columnas
    print("Columnas de X_train:", X_train.columns.tolist())

    # # # Inicio bloque MLflow # # #

    nombre_experimento = f"XGBoost_Optuna_study_{execution_date}"
    mlflow.set_experiment(nombre_experimento)
    mlflow.start_run(run_name="XGBoost_Optuna_Study_Global")

    # # # Fin bloque MLflow # # #

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # # # Inicio bloque MLflow # # #

    # Logueo de los mejores hiperparámetros en MLflow
    best_params = study.best_trial.params
    mlflow.log_params(best_params)

    # Guardado de gráficos de Optuna
    # Crea folder plots si no existe
    optuna.visualization.plot_optimization_history(study).write_image("optimization_history.png")
    mlflow.log_artifact("optimization_history.png", artifact_path="plots")
    # delete local file
    os.remove("optimization_history.png")

    optuna.visualization.plot_param_importances(study).write_image("param_importances.png")
    mlflow.log_artifact("param_importances.png", artifact_path="plots")
    # delete local file
    os.remove("param_importances.png")

    # Get best model from mlflow
    best_model_pipeline = get_best_model(mlflow.get_experiment_by_name(nombre_experimento).experiment_id)

    # Entrenamiento con los datos de entrenamiento completos
    y_val_pred = best_model_pipeline.predict(X_val)
    valid_recall = (y_val_pred[y_val == 1] == 1).mean()
    mlflow.log_metric("valid_recall", valid_recall)

    # Pickle save best model to models folder in mlruns
    os.makedirs("models", exist_ok=True)
    model_name = f"best_model_pipeline_{execution_date}.pkl"
    with open(f"models/{model_name}", "wb") as f:
        dump(best_model_pipeline, f)

    # Importancia de características con SHAP barras summary plot
    explainer = shap.TreeExplainer(best_model_pipeline["model"])
    X_train_date_transformed = best_model_pipeline["date_transformer"].transform(X_train)
    X_train_preprocessed = best_model_pipeline["preprocessor"].transform(X_train_date_transformed)
    shap_values = explainer(X_train_preprocessed)
    shap.summary_plot(
        shap_values,
        features=X_train_preprocessed,
        feature_names=X_train_preprocessed.columns,
        plot_type="bar",
        show=False,
    )
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png", artifact_path="plots")
    # delete local file
    os.remove("shap_summary.png")


def get_best_model(experiment_id):
    # Buscar la mejor corrida en MLflow basada en la métrica de validación
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_recall", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def model_predictor(**kwargs):
    """
    Función para cargar el modelo entrenado y realizar predicciones en los datos de la semana siguiente.
    """
    # Obtener la fecha desde Airflow o usar la actual
    execution_date = kwargs.get("ds")

    # Definir rutas
    base_folder = execution_date
    feature_extracted_folder = os.path.join(base_folder, "feature_extracted")
    models_folder = "models"
    predictions_folder = os.path.join(base_folder, "predictions")

    # Leer dataset de la semana siguiente con lags
    next_week_df = pd.read_parquet(os.path.join(feature_extracted_folder, "next_week_data.parquet"))

    # Obtener lista de mejores modelos guardados
    model_files = [f for f in os.listdir(models_folder) if f.startswith("best_model_pipeline_") and f.endswith(".pkl")]
    execution_dates = [f[len("best_model_pipeline_") : -len(".pkl")] for f in model_files]
    execution_dates = [date for date in execution_dates if date <= execution_date]

    # Seleccionamos el último mejor modelo basado en la fecha de ejecución
    latest_date = max(execution_dates)

    model_path = os.path.join(models_folder, f"best_model_pipeline_{latest_date}.pkl")
    trained_pipeline = load(model_path)

    # Realizar predicciones
    predictions = trained_pipeline.predict(next_week_df)

    # Guardar predicciones
    next_week_df["predictions"] = predictions
    next_week_df = next_week_df[["customer_id", "product_id", "predictions"]]
    output_path = os.path.join(predictions_folder, "predictions_next_week.csv")
    next_week_df.to_csv(output_path, index=False)
