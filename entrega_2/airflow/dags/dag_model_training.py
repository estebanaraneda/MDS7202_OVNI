# Python encargado de orquestar el proceso ETL (Extract, Transform, Load)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from etl_functions import split_data, training_lagged_features
from model_functions import optimize_model


# Definición del DAG
with DAG(
    dag_id="model_training",
    start_date=datetime(2025, 11, 10),
    schedule_interval="0 13 * * 1",  # cada lunes a las 1:00 PM
    catchup=False,
) as dag:
    """
    DAG para el entrenamiento y optimización del modelo de machine learning.
    Este DAG está programado para ejecutarse una hora después del DAG de tratamiento de datos,
    asegurando que los datos preprocesados estén disponibles para el entrenamiento del modelo.
    """

    # 1. Marcador de inicio del pipeline
    start_pipeline = EmptyOperator(task_id="start_pipeline")

    # 2. Crear lags y features adicionales
    feature_engineering = PythonOperator(
        task_id="feature_engineering", python_callable=training_lagged_features, provide_context=True
    )

    # 3. Crear splits de entrenamiento, validación y prueba
    create_splits = PythonOperator(task_id="create_splits", python_callable=split_data, provide_context=True)

    # 4. Model optimization
    model_optimization = PythonOperator(
        task_id="model_optimization", python_callable=optimize_model, provide_context=True
    )

    # Definición del flujo de tareas
    (start_pipeline >> feature_engineering >> create_splits >> model_optimization)
