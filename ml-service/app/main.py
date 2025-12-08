import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from pathlib import Path
from app.utils import preprocess_image
from PIL import Image
import io

# Load class names
with open(Path(__file__).parent.parent / "classes.json") as f:
    CLASS_NAMES = json.load(f)

# Load ONNX model
model_path = Path(__file__).parent.parent / "cropdoctor_final_b0.onnx"
providers = ["CPUExecutionProvider"]  # use CPU; remove CUDA if no GPU
session = ort.InferenceSession(str(model_path), providers=providers)

app = FastAPI(title="CropDoctor ML Service", version="1.0")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess
    input_array = preprocess_image(image).astype(np.float32)

    # Inference
    ort_inputs = {session.get_inputs()[0].name: input_array}
    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0]

    # Softmax + top-3
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    probs = probs[0]

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [
        {"disease": CLASS_NAMES[i], "confidence": float(probs[i])}
        for i in top3_idx
    ]

    return JSONResponse({
        "disease": top3[0]["disease"],
        "confidence": float(top3[0]["confidence"]),
        "top3": top3
    })
