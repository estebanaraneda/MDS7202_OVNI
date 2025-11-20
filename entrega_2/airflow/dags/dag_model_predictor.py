# Python encargado de orquestar el proceso ETL (Extract, Transform, Load)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from etl_functions import next_week_data
from model_functions import model_predictor


# Definición del DAG
with DAG(
    dag_id="model_predictor",
    start_date=datetime(2025, 11, 10),
    schedule_interval="0 14 * * 1",  # cada lunes a las 2:00 PM
    catchup=False,
) as dag:
    """
    DAG para el entrenamiento y optimización del modelo de machine learning.
    Este DAG está programado para ejecutarse una hora después del DAG de tratamiento de datos,
    asegurando que los datos preprocesados estén disponibles para el entrenamiento del modelo.
    """

    # 1. Marcador de inicio del pipeline
    start_pipeline = EmptyOperator(task_id="start_pipeline")

    # 2. Generar los datos para la semana que viene
    next_week_data_task = PythonOperator(task_id="next_week_data", python_callable=next_week_data, provide_context=True)

    # 3. Model predictions
    model_predictions = PythonOperator(
        task_id="model_predictions", python_callable=model_predictor, provide_context=True
    )

    # Definición del flujo de tareas
    (start_pipeline >> next_week_data_task >> model_predictions)
