# Python encargado de orquestar el proceso ETL (Extract, Transform, Load)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from etl_functions import create_folders, transform_data, split_data, lagged_features
from model_pipeline import optimize_model

# Definición del DAG
with DAG(
    dag_id="data_etl",
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,  # ejecución manual
    catchup=False,  # sin backfill
) as dag:

    # 1. Marcador de inicio del pipeline
    start_pipeline = EmptyOperator(task_id="start_pipeline")

    # 2. Crear carpetas de ejecución
    create_folders_task = PythonOperator(task_id="create_folders", python_callable=create_folders, provide_context=True)

    # 3. "Descargar datos históricos"
    # (Simula producción, en realidad solo copia archivos locales de base_data a raw en cada ejecución)
    download_data = BashOperator(
        task_id="download_data",
        bash_command="""
            mkdir -p {{ ti.xcom_pull(task_ids='create_folders')['raw_data_path'] }} && \
            cp $AIRFLOW_HOME/base_data/clientes.parquet \
                {{ ti.xcom_pull(task_ids='create_folders')['raw_data_path'] }}/clientes.parquet && \
            cp $AIRFLOW_HOME/base_data/productos.parquet \
                {{ ti.xcom_pull(task_ids='create_folders')['raw_data_path'] }}/productos.parquet && \
            cp $AIRFLOW_HOME/base_data/transacciones.parquet \
                {{ ti.xcom_pull(task_ids='create_folders')['raw_data_path'] }}/transacciones.parquet
    """,
    )

    # 4. Transformar y guardar datos semanalmente
    transform_data = PythonOperator(task_id="transform_data", python_callable=transform_data, provide_context=True)

    # 5. Crear lagged features y guardar datos
    feature_engineering = PythonOperator(
        task_id="feature_engineering", python_callable=lagged_features, provide_context=True
    )

    # 6. Crear splits de entrenamiento, validación y prueba
    create_splits = PythonOperator(task_id="create_splits", python_callable=split_data, provide_context=True)

    # 7. Model optimization
    model_optimization = PythonOperator(
        task_id="model_optimization", python_callable=optimize_model, provide_context=True
    )
