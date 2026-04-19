FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for Google Drive downloads
RUN pip install gdown

# opencv-python (GUI) requiere libGL.so.1 — no existe en slim y rompe import cv2.
# Alguna dependencia puede instalar opencv-python; dejar solo headless al final.
RUN pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir "opencv-python-headless==4.8.1.78"

# Copy application code
COPY . .

# Download quantized model checkpoint from Google Drive (gdown 5.x: --id removed; use URL + -O)
RUN if [ ! -f "checkpoint-quantized-model.pth" ]; then \
    echo "Downloading quantized model checkpoint from Google Drive using gdown..." && \
    ( gdown "https://drive.google.com/uc?id=1E6zIEuqAw7s6yl3st3N_jt-fNLfCegAO" -O checkpoint-quantized-model.pth \
      || gdown --fuzzy "https://drive.google.com/file/d/1E6zIEuqAw7s6yl3st3N_jt-fNLfCegAO/view?usp=sharing" -O checkpoint-quantized-model.pth ) && \
    echo "Quantized model checkpoint downloaded successfully"; \
    fi

# Download 64x3-CNN SavedModel from GitHub
RUN if [ ! -f "model-folder/64x3-CNN.model/saved_model.pb" ]; then \
    echo "Downloading 64x3-CNN.model from GitHub..." && \
    mkdir -p model-folder/64x3-CNN.model/variables && \
    BASE="https://raw.githubusercontent.com/Cyber-Aju/Diabetic_retinopathy/main/64x3-CNN.model" && \
    curl -fsSL "$BASE/fingerprint.pb"    -o model-folder/64x3-CNN.model/fingerprint.pb && \
    curl -fsSL "$BASE/keras_metadata.pb" -o model-folder/64x3-CNN.model/keras_metadata.pb && \
    curl -fsSL "$BASE/saved_model.pb"    -o model-folder/64x3-CNN.model/saved_model.pb && \
    curl -fsSL "$BASE/variables/variables.index" -o model-folder/64x3-CNN.model/variables/variables.index && \
    curl -fsSL "$BASE/variables/variables.data-00000-of-00001" -o model-folder/64x3-CNN.model/variables/variables.data-00000-of-00001 && \
    echo "64x3-CNN model downloaded successfully"; \
    fi

# Create directories
RUN mkdir -p /app/diabetic_retinopathy_dataset

# Expose port
EXPOSE 8000

# Cloud Run inyecta PORT; local suele usar 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]