from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model Keras saat server start
model = load_model("model_klasifikasi_sampah.keras")

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
