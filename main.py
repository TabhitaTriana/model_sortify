from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import io
import os
from PIL import Image
from tensorflow.keras.models import load_model
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Waste Classification API", version="1.0.0")

# Model loading dengan error handling
model = None
try:
    model_path = "model_klasifikasi_sampah.keras"
    
    # Debug: Print current directory and files
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully!")
    
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict

# Nama kelas - pastikan sesuai dengan model Anda
CLASS_NAMES = [
    "cardboard",    # 0
    "glass",        # 1
    "metal",        # 2
    "paper",        # 3
    "plastic",      # 4
    "shoes",        # 5
    "biological",   # 6
    "battery",      # 7
]

def preprocess_image(image: Image.Image):
    """Preprocess gambar sesuai dengan training model"""
    try:
        # Resize ke ukuran yang diharapkan model (sesuaikan dengan model Anda)
        image = image.resize((224, 224))
        
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
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.get("/")
def root():
    return {
        "message": "Waste Classification API",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "endpoints": ["/predict", "/model-info", "/health"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_waste(file: UploadFile = File(...)):
    """Prediksi jenis sampah dari gambar yang diupload"""
    
    # Validasi model
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia")
    
    # Validasi file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    
    # Validasi ukuran file (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file, 'size') and file.size and file.size > max_size:
        raise HTTPException(status_code=400, detail="File terlalu besar (maksimal 10MB)")
    
    try:
        # Baca gambar
        image_bytes = await file.read()
        
        # Validasi apakah file benar-benar gambar
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
            # Reopen after verify (verify() consumes the image)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as img_error:
            logger.error(f"Invalid image file: {str(img_error)}")
            raise HTTPException(status_code=400, detail="File bukan gambar yang valid")
        
        # Preprocess gambar
        processed_image = preprocess_image(image)
        
        # Prediksi
        predictions = model.predict(processed_image, verbose=0)  # verbose=0 untuk mengurangi log
        predictions = predictions[0]  # Remove batch dimension
        
        # Validasi jumlah kelas
        if len(predictions) != len(CLASS_NAMES):
            logger.error(f"Mismatch: Model output {len(predictions)} classes, but CLASS_NAMES has {len(CLASS_NAMES)}")
            raise HTTPException(status_code=500, detail="Model dan CLASS_NAMES tidak sesuai")
        
        # Get kelas dengan probabilitas tertinggi
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Semua probabilitas
        all_probabilities = {
            CLASS_NAMES[i]: float(predictions[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probabilities
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error dalam prediksi: {str(e)}")

@app.get("/model-info")
def model_info():
    """Get informasi tentang model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia")
    
    try:
        return {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "classes": CLASS_NAMES,
            "total_classes": len(CLASS_NAMES),
            "model_summary": {
                "total_params": model.count_params(),
                "layers": len(model.layers)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Exception handler untuk debugging
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)