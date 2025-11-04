from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python_operator import BranchPythonOperator
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF



# 🔹 Importar las funciones definidas en tu script anterior
# Asegúrate de reemplazar `pipeline_functions` por el nombre real del archivo .py donde están tus funciones
from hiring_dynamic_functions import create_folders, split_data, train_model, load_ands_merge, evaluate_models


# ------------------------------------------------------------
# 0. Definición del DAG
# ------------------------------------------------------------
with DAG(
    dag_id="hiring_paralel",
    start_date=datetime(2024, 10, 1),
    schedule_interval="0 15 5 * *",  # ejecución manual
    catchup=True,  # con backfill
    tags=["hiring", "paralel", "pipeline"],
) as dag:
    # 1. Marcador de inicio del pipeline
    start_pipeline = EmptyOperator(task_id="start_pipeline")

    # 2. Crear carpetas de ejecución
    create_folders_task = PythonOperator(task_id="create_folders", python_callable=create_folders, provide_context=True)

    #branching
    def choose_branch(**kwargs):
        if datetime.now(timezone.utc) < datetime(2024,10,1):
            return 'branch_a'
        else:
            return 'branch_b'
    
    branch_task = BranchPythonOperator(
    task_id='branch_task',
    python_callable=choose_branch,
    provide_context=True,
    dag=dag)

    # 3. Descargar dataset y guardarlo en carpeta 'raw'
    # Reemplaza test_url con la URL real cuando la tengas
    execution_date = "{{ ds }}"
    download_data_a = BashOperator(
        task_id="download_data_a",
        bash_command=(
            f"curl -s -o $AIRFLOW_HOME/{execution_date}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )
    
    download_data_b = BashOperator(
        task_id="download_data_b",
        bash_command=(
            f"curl -s -o $AIRFLOW_HOME/{execution_date}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
            f" && curl -s -o $AIRFLOW_HOME/{execution_date}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv",
    ))
    
    # 4. Merge
    merge_data_task = PythonOperator(task_id="merge_data", python_callable=load_ands_merge, trigger_rule='one_success', provide_context=True)
    
    # 5. Aplicar hold-out split
    split_data_task = PythonOperator(task_id="split_data", python_callable=split_data, provide_context=True)



    # 6.1 RF
    rf_model_task = PythonOperator(
        task_id="rf_model", python_callable=train_model, op_args=[RandomForestClassifier], provide_context=True
    )
        # 6.2 modelo2
    m2_model_task = PythonOperator(
        task_id="m2_model", python_callable=train_model, op_args=[SVC], provide_context=True
    )
        #6.3 modelo3
    m3_model_task = PythonOperator(
        task_id="m3_model", python_callable=train_model,op_args=[RBF], provide_context=True
    )
    # 7.
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",python_callable=evaluate_models, provide_context=True , trigger_rule='none_failed'
    )

    # Definir flujo lineal de tareas
    (
        start_pipeline
        >> create_folders_task 
        >> branch_task 
        >> [download_data_a, download_data_b]
        >> split_data_task
        >> [rf_model_task, m2_model_task, m3_model_task]
        >> evaluate_model_task
    )
