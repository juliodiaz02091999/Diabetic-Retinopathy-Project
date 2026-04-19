from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os
import threading
import numpy as np
import io
from pydantic import BaseModel
import tempfile
import shutil

app = FastAPI(
    title="RetinaScan AI API",
    description="Advanced AI-Powered Diabetic Retinopathy Screening Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes - útil para desarrollo y testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML API Configuration

# Pydantic Models for ML responses only

class PredictionResponse(BaseModel):
    confidence_score: float
    prediction_class: str
    diagnosis: str
    model_used: str = "Current Model"

class CNNPredictionResponse(BaseModel):
    confidence_score: float
    prediction_class: str
    diagnosis: str
    probabilities: dict
    clinical_recommendation: str
    model_used: str
    model_loaded: bool

class RETFoundPredictionResponse(BaseModel):
    # Interpretación principal (binaria para screening)
    confidence_score: float
    prediction_class: str
    diagnosis: str
    probabilities: dict
    
    # Interpretación detallada (clase individual más probable)
    individual_prediction: str
    individual_confidence: float
    individual_diagnosis: str
    
    # Interpretación binaria explícita
    binary_prediction: str
    binary_confidence: float
    binary_diagnosis: str
    
    # Recomendación clínica
    clinical_recommendation: str
    
    # Información detallada (compatibilidad)
    detailed_class: str
    detailed_probabilities: dict
    model_used: str
    checkpoint_loaded: bool

# ── Models: load in background so uvicorn binds PORT before Cloud Run timeout ──
model = None
retfound_model = None
cnn_model = None
_models_ready = False


def _load_all_models():
    global model, retfound_model, cnn_model, _models_ready
    try:
        import tensorflow as tf
        from retfound_official import RETFoundOfficial
        from cnn_model import CyberAjuCNN

        # 1) CNN first — /predict/cnn usable pronto; RETFound en CPU tarda mucho
        try:
            cnn_model = CyberAjuCNN()
            if not cnn_model.model_loaded:
                raise RuntimeError("Model failed to initialise")
            print("✅ 64x3-CNN model loaded!")
        except Exception as e:
            print(f"❌ Error loading 64x3-CNN model: {e}")
            cnn_model = None

        try:
            model = tf.keras.models.load_model("model-folder/diabetic-retino-model.h5")
            print("✅ Current model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading current model: {e}")
            model = None

        try:
            checkpoint_path = "checkpoint-quantized-model.pth"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            retfound_model = RETFoundOfficial(checkpoint_path=checkpoint_path)
            print(f"✅ RETFound loaded from {checkpoint_path}!")
        except Exception as e:
            print(f"❌ Error loading RETFound: {e}")
            retfound_model = None

        print("✅ Model loading pass finished.")
    except Exception as e:
        print(f"❌ Fatal error in model loader: {e}")
    finally:
        _models_ready = True


threading.Thread(target=_load_all_models, daemon=True).start()

# ML API ready - Database and auth moved to Supabase

# API Endpoints

@app.get("/")
async def root():
    return {"message": "RetinaScan AI API - Diabetic Retinopathy Screening Platform"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RetinaScan AI API",
        "models_ready": _models_ready,
        "models": {
            "current_model": "available" if model else ("loading" if not _models_ready else "unavailable"),
            "retfound_quantized_model": "available" if retfound_model else ("loading" if not _models_ready else "unavailable"),
            "quantized_checkpoint_exists": os.path.exists("checkpoint-quantized-model.pth"),
            "cnn_model": "available" if cnn_model else ("loading" if not _models_ready else "unavailable"),
        },
    }

# ML endpoints only - Auth and patient management moved to Supabase

@app.post("/predict", response_model=PredictionResponse)
async def predict_retinopathy(
    file: UploadFile = File(...)
):
    if not model:
        detail = "Model is still loading, retry shortly" if not _models_ready else "Model not loaded"
        raise HTTPException(status_code=503, detail=detail)
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Read and save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        from model import predict_image

        confidence_level = predict_image(model=model, image_path=temp_file_path)
        confidence_score = float(confidence_level[0, 0]) * 100
        
        # Determine diagnosis
        if confidence_level >= 0.5:
            prediction_class = "NO-DR"
            diagnosis = "Negative for Diabetic Retinopathy"
        else:
            prediction_class = "DR"
            diagnosis = "Positive for Diabetic Retinopathy"
            confidence_score = 100 - confidence_score
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return PredictionResponse(
            confidence_score=confidence_score,
            prediction_class=prediction_class,
            diagnosis=diagnosis,
            model_used="Current Model (.h5)"
        )
        
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/retfound", response_model=RETFoundPredictionResponse)
async def predict_retinopathy_retfound(
    file: UploadFile = File(...)
):
    if not retfound_model:
        detail = "RETFound is still loading, retry shortly" if not _models_ready else "RETFound model not loaded"
        raise HTTPException(status_code=503, detail=detail)
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Read and save uploaded file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Predict using RETFound
        result = retfound_model.predict(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return RETFoundPredictionResponse(
            # Interpretación principal (binaria para screening)
            confidence_score=result['confidence_score'],
            prediction_class=result['prediction_class'],
            diagnosis=result['diagnosis'],
            probabilities=result['probabilities'],
            
            # Interpretación detallada (clase individual más probable)
            individual_prediction=result['individual_prediction'],
            individual_confidence=result['individual_confidence'],
            individual_diagnosis=result['individual_diagnosis'],
            
            # Interpretación binaria explícita
            binary_prediction=result['binary_prediction'],
            binary_confidence=result['binary_confidence'],
            binary_diagnosis=result['binary_diagnosis'],
            
            # Recomendación clínica
            clinical_recommendation=result['clinical_recommendation'],
            
            # Información detallada (compatibilidad)
            detailed_class=result['detailed_class'],
            detailed_probabilities=result['detailed_probabilities'],
            model_used=result['model_used'],
            checkpoint_loaded=result['checkpoint_loaded']
        )
        
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"RETFound prediction failed: {str(e)}")

@app.post("/predict/cnn", response_model=CNNPredictionResponse)
async def predict_retinopathy_cnn(
    file: UploadFile = File(...)
):
    if not cnn_model:
        detail = "CNN is still loading, retry shortly" if not _models_ready else "64x3-CNN model not loaded"
        raise HTTPException(status_code=503, detail=detail)

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = cnn_model.predict(temp_file_path)
        os.unlink(temp_file_path)

        return CNNPredictionResponse(**result)

    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {str(e)}")


# Prediction management moved to Supabase - only ML endpoints remain

@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    return {
        "current_model": {
            "name": "Current Model (.h5)",
            "status": "available" if model else "unavailable",
            "description": "Original diabetic retinopathy model",
            "endpoint": "/predict"
        },
        "retfound_quantized_model": {
            "name": "RETFound Official (Quantized)",
            "status": "available" if retfound_model else "unavailable", 
            "description": "Foundation model for retinal imaging (Nature 2023) - Quantized version",
            "endpoint": "/predict/retfound",
            "checkpoint_loaded": getattr(retfound_model, 'checkpoint_loaded', False) if retfound_model else False
        },
        "cnn_model": {
            "name": "64x3-CNN (Cyber-Aju)",
            "status": "available" if cnn_model else "unavailable",
            "description": "CNN-based DR detection model (>93% accuracy) — SavedModel format",
            "endpoint": "/predict/cnn",
            "model_loaded": getattr(cnn_model, 'model_loaded', False) if cnn_model else False
        }
    }

@app.post("/preprocess/preview")
async def preprocess_preview(file: UploadFile = File(...)):
    """Returns the Gaussian-filtered + resized image that the 64x3-CNN model receives before inference."""
    import cv2

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        content = await file.read()
        nparr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX=10), -4, 128)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_bgr)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)