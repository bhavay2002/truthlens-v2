import torch
import gradio as gr

# 🔴 Import your custom model class
from src.models.multitask.multitask_truthlens_model import MultiTaskTruthLensModel

# 🔹 Load model
model = torch.load("model.pt", map_location="cpu", weights_only=False)
model.eval()

# 🔹 Dummy preprocess (replace if needed)
def preprocess(text):
    return text

# 🔹 Prediction
def predict(text):
    try:
        # ⚠️ Replace with your real inference pipeline
        output = model(preprocess(text))

        return {
            "result": str(output)
        }

    except Exception as e:
        return {"error": str(e)}

# 🔹 Gradio
gr.Interface(
    fn=predict,
    inputs="text",
    outputs="json",
    title="TruthLens AI"
).launch()