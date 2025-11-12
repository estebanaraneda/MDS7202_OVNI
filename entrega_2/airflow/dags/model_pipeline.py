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
from joblib import dump


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

    # Definir variable objetivo
    target_col = "order"

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
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

    # Entrenamos el modelo
    model_to_optimize.fit(X_train, y_train)
    # Realizamos las predicciones
    y_opt_val_pred = model_to_optimize.predict(X_val)
    # Calculamos la métrica de evaluación
    recall = (y_opt_val_pred[y_val == 1] == 1).mean()
    # Guardamos el mejor pipeline entrenado
    trial.set_user_attr("best_pipeline", model_to_optimize)
    return recall


def optimize_model(**kwargs):

    # Setamos X_train, y_train, X_val, y_val, X_test, y_test como variables globales
    # Esto para no tener que leerlas dentro de la función objective en cada iteración
    global X_train, y_train, X_val, y_val, X_test, y_test

    execution_date = kwargs.get("ds")
    X_train, y_train, X_val, y_val, X_test, y_test = split_reading_data(execution_date)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Número de ensayos realizados:", len(study.trials))
    print("Mejor valor de recall en validación:", study.best_value)
    print("Mejores hiperparámetros:", study.best_params)

    # Obtener el mejor pipeline entrenado
    best_pipeline = study.best_trial.user_attrs["best_pipeline"]

    # Evaluar en el conjunto de prueba
    y_test_pred = best_pipeline.predict(X_test)
    test_recall = (y_test_pred[y_test == 1] == 1).mean()
    print("Recall en conjunto de prueba:", test_recall)

    # Guardar el mejor pipeline entrenado
    base_folder = execution_date
    models_folder = os.path.join(base_folder, "models")
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, "best_pipeline.joblib")

    dump(best_pipeline, model_path)
