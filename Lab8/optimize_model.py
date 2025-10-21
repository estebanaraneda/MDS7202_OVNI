import pandas as pd
import xgboost as xgb

# pipelines
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
import pickle
import mlflow
import os
import shap
import matplotlib.pyplot as plt
import matplotlib


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def optimize_model():
    # Carga de datos
    data = pd.read_csv("water_potability.csv")

    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    # División de datos
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )  # 0.25 x 0.8 = 0.2

    # Función objetivo para Optuna
    def objective(trial):

        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

        # Preprocesamiento
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, X.columns),
            ]
        )

        # Pipeline completo
        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier())])

        # Inicio de una nueva corrida en MLflow
        mlflow.start_run(nested=True, run_name=f"trial_{trial.number}")

        # Entrenamiento y evaluación
        model_pipeline["classifier"].set_params(**param)
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)

        metrica = f1_score(y_val, y_pred)

        # Logueo de parámetros y métricas en MLflow
        mlflow.log_params(param)
        mlflow.log_metric("valid_f1", metrica)

        # Guardado del modelo en MLflow
        signature = mlflow.models.infer_signature(X_train, model_pipeline.predict(X_train))
        mlflow.sklearn.log_model(model_pipeline, "model", signature=signature, input_example=X_train.iloc[:5])

        # Finalización de la corrida en MLflow
        mlflow.end_run()

        return metrica

    # Inicialización de MLflow
    now_timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    nombre_experimento = f"XGBoost_Optuna_Water_Potability_{now_timestamp}"
    mlflow.set_experiment(nombre_experimento)
    mlflow.start_run(run_name="XGBoost_Optuna_Study_Global")

    # Optimización con Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Guardado del mejor modelo
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

    # Preprocesamiento
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, X.columns),
        ]
    )

    # Evaluación final en el conjunto de prueba
    best_model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier(**best_params))]
    )

    # Entrenamiento con los datos de entrenamiento completos
    best_model_pipeline.fit(X_train, y_train)
    y_val_pred = best_model_pipeline.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    mlflow.log_metric("valid_f1", val_f1)

    # Get best model from mlflow
    best_model_pipeline = get_best_model(mlflow.get_experiment_by_name(nombre_experimento).experiment_id)

    # Pickle save best model to models folder in mlruns
    os.makedirs("models", exist_ok=True)
    model_name = f"best_model_pipeline_{now_timestamp}.pkl"
    with open(f"models/{model_name}", "wb") as f:
        pickle.dump(best_model_pipeline, f)

    # Importancia de características con SHAP barras summary plot
    explainer = shap.Explainer(best_model_pipeline["classifier"])
    X_train_preprocessed = best_model_pipeline["preprocessor"].transform(X_train)
    shap_values = explainer(X_train_preprocessed)
    shap.summary_plot(shap_values, features=X_train_preprocessed, feature_names=X.columns, plot_type="bar", show=False)
    plt.savefig("shap_summary.png")
    mlflow.log_artifact("shap_summary.png", artifact_path="plots")
    # delete local file
    os.remove("shap_summary.png")

    # Gráfico de configuración final de hiperparámetros
    params_df = pd.DataFrame(list(best_params.items()), columns=["Hyperparameter", "Value"])

    # Dibujar tabla
    plt.figure(figsize=(8, len(params_df) * 0.3))
    plt.axis("off")
    plt.title("Configuraciones del modelo final", fontsize=12, fontweight="bold", pad=15)
    table = plt.table(cellText=params_df.values, colLabels=params_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    config_plot_path = "final_model_configurations.png"
    plt.savefig(config_plot_path, bbox_inches="tight")
    plt.close()

    # Loguear figura como artifact
    mlflow.log_artifact(config_plot_path, artifact_path="plots")
    os.remove(config_plot_path)

    mlflow.end_run()


if __name__ == "__main__":
    optimize_model()
