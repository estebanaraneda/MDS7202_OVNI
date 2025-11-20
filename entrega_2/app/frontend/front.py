import gradio as gr
import requests
import json

BACKEND_URL = "http://backend:8000/predict"


def predict_from_json(input_json: str):
    """
    Recibe un JSON completo como texto.
    Lo envía directamente al backend.
    """

    # Validar JSON
    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError:
        return "❌ Error: El texto ingresado no es un JSON válido."

    # Llamar al backend
    try:
        response = requests.post(BACKEND_URL, json=payload)
    except Exception as e:
        return f"❌ Error de conexión con backend: {e}"

    # Verificar respuesta
    if response.status_code != 200:
        return f"❌ Error del backend: {response.text}"

    return json.dumps(response.json(), indent=2, ensure_ascii=False)


# -------- GRADIO UI --------
iface = gr.Interface(
    fn=predict_from_json,
    inputs=gr.Textbox(label="JSON de entrada", lines=20, placeholder="Pegue aquí el JSON de la predicción"),
    outputs=gr.Textbox(label="Respuesta del backend", lines=20),
    title="Frontend de Predicción",
    description="Pegue un JSON completo con todas las variables.",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
