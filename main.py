from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os

app = FastAPI()

# Endpoint sederhana untuk cek API hidup
@app.get("/")
def root():
    return {"status": "API is running", "message": "Model is ready for prediction"}

print("Isi folder saat ini:", os.listdir())

# Load model
model = load_model("model_klasifikasi_sampah.keras")
class_names = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']  # ganti sesuai klasemu

# Preprocessing function
def preprocess_image(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalisasi
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img_array = preprocess_image(content)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2)
    })
