from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os

# Antes de importar TensorFlow (carga perezosa en hilo): evita explosión de hilos en Cloud Run con 1–2 vCPU
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import threading
import numpy as np
import io
from pydantic import BaseModel
import tempfile

app = FastAPI(
    title="RetinaScan AI API",
    description="Advanced AI-Powered Diabetic Retinopathy Screening Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CNNPredictionResponse(BaseModel):
    confidence_score: float
    prediction_class: str
    diagnosis: str
    probabilities: dict
    clinical_recommendation: str
    model_used: str
    model_loaded: bool


class RETFoundPredictionResponse(BaseModel):
    confidence_score: float
    prediction_class: str
    diagnosis: str
    probabilities: dict
    individual_prediction: str
    individual_confidence: float
    individual_diagnosis: str
    binary_prediction: str
    binary_confidence: float
    binary_diagnosis: str
    clinical_recommendation: str
    detailed_class: str
    detailed_probabilities: dict
    model_used: str
    checkpoint_loaded: bool


retfound_model = None
cnn_model = None
_models_ready = False
_retfound_init_done = False
_cnn_load_error: str | None = None
_retfound_load_error: str | None = None


def _load_retfound_only():
    global retfound_model, _retfound_init_done, _retfound_load_error
    _retfound_load_error = None
    try:
        from retfound_official import RETFoundOfficial

        checkpoint_path = "checkpoint-quantized-model.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        retfound_model = RETFoundOfficial(checkpoint_path=checkpoint_path)
        print(f"✅ RETFound loaded from {checkpoint_path}!")
    except Exception as e:
        err = str(e)
        print(f"❌ Error loading RETFound: {e}")
        retfound_model = None
        _retfound_load_error = err[:800]
    finally:
        _retfound_init_done = True


def _load_core_models():
    """Solo 64x3-CNN (TensorFlow SavedModel). RETFound se carga en otro hilo."""
    global cnn_model, _models_ready, _cnn_load_error
    _cnn_load_error = None
    print("[models] Core loader thread started", flush=True)
    try:
        print("[models] Importing CyberAjuCNN (loads TensorFlow)…", flush=True)
        from cnn_model import CyberAjuCNN

        try:
            print("[models] Loading SavedModel from disk (may take 1–3 min on small CPU)…", flush=True)
            cnn_model = CyberAjuCNN()
            if not cnn_model.model_loaded:
                raise RuntimeError("CyberAjuCNN.model_loaded is False (SavedModel load failed — see logs)")
            print("✅ 64x3-CNN model loaded!")
        except Exception as e:
            err = str(e)
            print(f"❌ Error loading 64x3-CNN model: {e}")
            cnn_model = None
            _cnn_load_error = err[:800]

        _models_ready = True
        print("✅ Core models pass finished.")
    except Exception as e:
        err = str(e)
        print(f"❌ Fatal error in core model loader: {e}")
        if _cnn_load_error is None:
            _cnn_load_error = err[:800]
        if not _models_ready:
            _models_ready = True

    threading.Thread(target=_load_retfound_only, daemon=True).start()


threading.Thread(target=_load_core_models, daemon=True).start()


@app.get("/")
async def root():
    return {"message": "RetinaScan AI API - Diabetic Retinopathy Screening Platform"}


@app.get("/health")
async def health_check():
    def _cnn_status() -> str:
        if cnn_model:
            return "available"
        if not _models_ready:
            return "loading"
        return "unavailable"

    def _retfound_status() -> str:
        if retfound_model:
            return "available"
        if not _retfound_init_done:
            return "loading"
        return "unavailable"

    return {
        "status": "healthy",
        "service": "RetinaScan AI API",
        "models_ready": _models_ready,
        "retfound_init_done": _retfound_init_done,
        "diagnostics": {
            "cnn_load_error": _cnn_load_error,
            "retfound_load_error": _retfound_load_error,
            "saved_model_pb_exists": os.path.isfile(
                os.path.join("model-folder", "64x3-CNN.model", "saved_model.pb")
            ),
        },
        "models": {
            "retfound_quantized_model": _retfound_status(),
            "quantized_checkpoint_exists": os.path.exists("checkpoint-quantized-model.pth"),
            "cnn_model": _cnn_status(),
        },
    }


@app.post("/predict/retfound", response_model=RETFoundPredictionResponse)
async def predict_retinopathy_retfound(
    file: UploadFile = File(...)
):
    if not retfound_model:
        detail = (
            "RETFound is still loading (ViT on CPU can take several minutes), retry shortly"
            if not _retfound_init_done
            else "RETFound model not loaded"
        )
        raise HTTPException(status_code=503, detail=detail)

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = retfound_model.predict(temp_file_path)
        os.unlink(temp_file_path)

        return RETFoundPredictionResponse(
            confidence_score=result['confidence_score'],
            prediction_class=result['prediction_class'],
            diagnosis=result['diagnosis'],
            probabilities=result['probabilities'],
            individual_prediction=result['individual_prediction'],
            individual_confidence=result['individual_confidence'],
            individual_diagnosis=result['individual_diagnosis'],
            binary_prediction=result['binary_prediction'],
            binary_confidence=result['binary_confidence'],
            binary_diagnosis=result['binary_diagnosis'],
            clinical_recommendation=result['clinical_recommendation'],
            detailed_class=result['detailed_class'],
            detailed_probabilities=result['detailed_probabilities'],
            model_used=result['model_used'],
            checkpoint_loaded=result['checkpoint_loaded']
        )

    except Exception as e:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"RETFound prediction failed: {str(e)}")


@app.post("/predict/cnn", response_model=CNNPredictionResponse)
async def predict_retinopathy_cnn(
    file: UploadFile = File(...)
):
    if not cnn_model:
        if not _models_ready:
            detail = "CNN is still loading, retry shortly"
        elif _cnn_load_error:
            detail = f"64x3-CNN failed to load: {_cnn_load_error}"
        else:
            detail = "64x3-CNN model not loaded"
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
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {str(e)}")


@app.get("/models/info")
async def get_models_info():
    return {
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
