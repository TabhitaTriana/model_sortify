from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import gdown

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running", "message": "Model is ready for prediction"}

# === Download model from Google Drive if not exists ===
model_path = "model_klasifikasi_sampah.h5"
drive_url = "https://drive.google.com/uc?id=14N_orVJnO047XIJvqs7GMx3EHjPlNtLH"  # Ganti dengan ID Google Drive kamu

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(drive_url, model_path, quiet=False)

# === Load the H5 model ===
model = load_model(model_path)

# === Class Names ===
class_names = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']

# === Image Preprocessing ===
def preprocess_image(file) -> np.ndarray:
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Prediction Endpoint ===
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
