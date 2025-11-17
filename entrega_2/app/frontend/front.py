# frontend/app.py

import gradio as gr
import requests

BACKEND_URL = "http://backend:8000/prediction"

def predict(text):
    try:
        response = requests.post(BACKEND_URL, json={"text": text})
        return response.json()
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks() as ui:
    gr.Markdown("##Predicción de compra")
    input1 = gr.Textbox(label="Entrada")
    
    dag_name_input = gr.Textbox(label="DAG ID", value="example_dag")
    trigger_btn = gr.Button("Triggerpreddicion DAG")
    output = gr.Textbox(label="Result")
    
    trigger_btn.click(lambda dag_id: trigger_dag_run(), inputs=dag_name_input, outputs=output)


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)