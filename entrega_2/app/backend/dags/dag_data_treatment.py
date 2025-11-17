# Python encargado de orquestar el proceso ETL (Extract, Transform, Load)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from etl_functions import create_folders, transform_data
import os
from airflow.operators.python import BranchPythonOperator
import logging

# Definición del DAG
with DAG(
    dag_id="data_treatment",
    start_date=datetime(2025, 10, 14),
    schedule_interval="0 12 * * 1",  # cada lunes a las 12:00 PM
    catchup=True,
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
    download_new_data = BashOperator(
        task_id="download_new_data",
        bash_command="""
            cp $AIRFLOW_HOME/base_data/new_transactions.parquet \
                {{ ti.xcom_pull(task_ids='create_folders')['raw_data_path'] }}/new_transactions.parquet
    """,
    )

    # 4. Preprocesamos todos los datos disponibles o utilizamos los datos preprocesados hasta la semana pasada
    # y solo preprocesamos los datos nuevos si es que existen.
    last_week_preprocessed_path = "{{ macros.ds_add(ds, -7) }}/preprocessed/weekly_data.parquet"

    def choose_branch(last_week_preprocessed_path=None):
        logger = logging.getLogger("airflow.task")
        logger.info("Verificando si existe el archivo de la semana pasada...")
        logger.info(f"Ruta: {last_week_preprocessed_path}")
        # Si existe el archivo de la semana pasada significa que ya existen datos preprocesados
        # Por lo que no es necesario realizar el preprocesamiento entero de nuevo.
        if os.path.exists(last_week_preprocessed_path):
            return "preprocess_from_last_week"
        else:
            return "preprocess_full"

    branch_task = BranchPythonOperator(
        task_id="branch_task",
        python_callable=choose_branch,
        dag=dag,
        op_kwargs={"last_week_preprocessed_path": last_week_preprocessed_path},
    )

    preprocess_full = PythonOperator(
        task_id="preprocess_full",
        python_callable=transform_data,
        op_kwargs={"only_last_week": False},
        provide_context=True,
    )

    preprocess_from_last_week = PythonOperator(
        task_id="preprocess_from_last_week",
        python_callable=transform_data,
        op_kwargs={"only_last_week": True, "last_week_path": last_week_preprocessed_path},
        provide_context=True,
    )

    # Definición de la secuencia de tareas
    (
        start_pipeline
        >> create_folders_task
        >> download_data
        >> download_new_data
        >> branch_task
        >> [preprocess_full, preprocess_from_last_week]
    )
