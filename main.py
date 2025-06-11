from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import os
from pathlib import Path

app = FastAPI()

# Load model Keras saat server start
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_klasifikasi_sampah.keras"

print(f"Looking for model at: {MODEL_PATH}")
assert MODEL_PATH.exists(), "❌ Model file not found!"

model = load_model(MODEL_PATH)
# Sesuaikan input dengan fitur model kamu
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
