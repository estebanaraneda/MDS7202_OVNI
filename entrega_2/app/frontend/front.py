# frontend/app.py

import gradio as gr
import requests

BACKEND_URL = "http://backend:8000/prediction"
AIRFLOW_USERNAME = "admin"
AIRFLOW_PASSWORD = "admin"

def predict(text):
    try:
        response = requests.post(BACKEND_URL, json={"text": text})
        return response.json()
    except Exception as e:
        return {"error": str(e)}
def trigger_dag_run():
    dag_id="dag_model_predictor"
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    payload = {
        "dag_run_id": f"manual__{dag_id}__{{{{ ts_nodash }}}}",
        "conf": {"source": "gradio-ui"},   # let Airflow use execution_date = now
        "note": "Triggered from Gradio UI"}
    try:
        response = requests.post(
            url,
            json=payload,
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
        )
        return f"Status {response.status_code}: {response.text}"
    except Exception as e:
        return str(e)
    
with gr.Blocks(title="Predicción + Airflow Trigger") as ui:
    gr.Markdown("# Predicción de compra")
    
    with gr.Row():
    #    with gr.Column(scale=1):
    #        gr.Markdown("## 1. Modelo de predicción")
    #        input_text = gr.Textbox(label="Texto de entrada", lines=3)
    #        predict_btn = gr.Button("Predecir")
    #        prediction_output = gr.JSON(label="Resultado predicción")
            
        with gr.Column(scale=1):
            gr.Markdown("Activar la predicción")
            trigger_btn = gr.Button("Activar DAG", variant="primary")
            trigger_output = gr.Textbox(label="Resultado")
    with gr.Row():
            gr.Textbox(
            value='''Esta interfás ocupa el botón de "activar la predicción" para iniciar la predicción de los datos. Y lo devuelve en la sección de resultados.
            ''',
            label="Instrucciones de uso",
            interactive=False,
            lines=3
            )

    #predict_btn.click(predict, inputs=input_text, outputs=prediction_output)
    trigger_btn.click(trigger_dag_run, inputs=None, outputs=trigger_output)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)