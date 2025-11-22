# Python encargado de orquestar el proceso ETL (Extract, Transform, Load)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from etl_functions import create_folders, transform_data
import os

# Definición del DAG
with DAG(
    dag_id="data_treatment",
    start_date=datetime(2025, 11, 10),
    schedule_interval="0 12 * * 1",  # cada lunes a las 12:00 PM
    catchup=False,
    max_active_runs=1,
    tags=["data_treatment", "etl", "weekly_pipeline"],
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

    # 4. "Descargar datos nuevos"
    # (Simula producción, en realidad solo copia archivos locales de new_data a raw en cada ejecución)
    # Mueve la carpeta entera base_data/new_transactions/ a la carpeta raw del día de ejecución
    download_new_data = BashOperator(
        task_id="download_new_data",
        bash_command="""
            cp -r $AIRFLOW_HOME/base_data/new_transactions/ \
                {{ ti.xcom_pull(task_ids='create_folders')['raw_data_path'] }}/new_transactions
        """,
    )

    home_folder = os.environ["AIRFLOW_HOME"]

    preprocess_from_last_execution = PythonOperator(
        task_id="preprocess_data",
        python_callable=transform_data,
        op_kwargs={"home_folder": home_folder},
        provide_context=True,
    )

    # Definición de la secuencia de tareas
    (start_pipeline >> create_folders_task >> download_data >> download_new_data >> preprocess_from_last_execution)
