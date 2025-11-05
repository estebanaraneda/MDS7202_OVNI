from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# Asegúrate de reemplazar `pipeline_functions` por el nombre real del archivo .py donde están tus funciones
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface


# ------------------------------------------------------------
# 0. Definición del DAG
# ------------------------------------------------------------
with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,  # ejecución manual
    catchup=False,  # sin backfill
    tags=["hiring", "lineal", "pipeline"],
) as dag:
    # 1. Marcador de inicio del pipeline
    start_pipeline = EmptyOperator(task_id="start_pipeline")

    # 2. Crear carpetas de ejecución
    create_folders_task = PythonOperator(task_id="create_folders", python_callable=create_folders, provide_context=True)

    # 3. Descargar dataset y guardarlo en carpeta 'raw'
    # Reemplaza test_url con la URL real cuando la tengas
    execution_date = "{{ ds }}"
    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            f"curl -s -o $AIRFLOW_HOME/{execution_date}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )

    # 4. Aplicar hold-out split
    split_data_task = PythonOperator(task_id="split_data", python_callable=split_data, provide_context=True)

    # 5. Preprocesar y entrenar modelo
    preprocess_and_train_task = PythonOperator(
        task_id="preprocess_and_train", python_callable=preprocess_and_train, provide_context=True
    )

    # 6. Montar interfaz en Gradio
    gradio_interface_task = PythonOperator(
        task_id="gradio_interface", python_callable=gradio_interface, provide_context=True
    )

    # Definir flujo lineal de tareas
    (
        start_pipeline
        >> create_folders_task
        >> download_data
        >> split_data_task
        >> preprocess_and_train_task
        >> gradio_interface_task
    )
