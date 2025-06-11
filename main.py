from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model Keras saat server start
model = load_model("model_klasifikasi_sampah.keras")

# Response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict

# Nama kelas - sesuaikan dengan model Anda
CLASS_NAMES = [
    "cardboard",    # 0
    "glass",        # 1
    "metal",        # 2
    "paper",        # 3
    "plastic",      # 4
    "shoes",        # 5
    "biological",
    "battery",
]
# GANTI dengan kelas yang sesuai model Anda, misalnya:
# CLASS_NAMES = ["organik", "anorganik", "daur_ulang"]

def preprocess_image(image: Image.Image):
    """Preprocess gambar sesuai dengan training model"""
    # Resize ke ukuran yang diharapkan model (biasanya 224x224 atau 150x150)
    image = image.resize((224, 224))  # Sesuaikan dengan input size model Anda
    
    # Convert ke RGB jika perlu
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert ke array numpy
    image_array = np.array(image)
    
    # Normalisasi pixel values ke 0-1
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.get("/")
def root():
    return {"message": "Waste Classification API", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_waste(file: UploadFile = File(...)):
    """Prediksi jenis sampah dari gambar yang diupload"""
    
    # Validasi file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    
    try:
        # Baca gambar
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess gambar
        processed_image = preprocess_image(image)
        
        # Prediksi
        predictions = model.predict(processed_image)
        predictions = predictions[0]  # Remove batch dimension
        
        # Get kelas dengan probabilitas tertinggi
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Semua probabilitas
        all_probabilities = {
            CLASS_NAMES[i]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam prediksi: {str(e)}")

# Endpoint tambahan untuk debugging
@app.get("/model-info")
def model_info():
    """Get informasi tentang model"""
    return {
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES)
    }