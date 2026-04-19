FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (git-lfs alineado con flujo LFS en repo; en build se usan curl como en TORTUOSITY)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    git \
    git-lfs \
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

# Copy application code (con Git LFS, aquí entran punteros en model-folder/)
COPY . .

# 64x3-CNN: patrón TORTUOSITY — si hay .gitattributes + LFS, se bajan los binarios con curl
# desde TU repo (raw.githubusercontent.com), no desde terceros.
# Ajusta CNN_RAW_BASE si tu usuario/repo/rama/ruta difieren (o pasa --build-arg en CI).
ARG CNN_RAW_BASE=https://raw.githubusercontent.com/juliodiaz02091999/Diabetic-Retinopathy-Project/main/backend/model-folder/64x3-CNN.model

RUN mkdir -p model-folder/64x3-CNN.model/variables && \
    P="model-folder/64x3-CNN.model/saved_model.pb" && \
    NEED=0 && \
    if [ ! -f "$P" ]; then NEED=1; \
    elif [ -f ".gitattributes" ] && grep -q "filter=lfs" .gitattributes && grep -q "model-folder/64x3-CNN" .gitattributes && [ "$(wc -c < "$P" | tr -d ' ')" -lt 2048 ]; then NEED=1; \
    fi && \
    if [ "$NEED" -eq 1 ]; then \
      if [ ! -f ".gitattributes" ] || ! grep -q "filter=lfs" .gitattributes || ! grep -q "model-folder/64x3-CNN" .gitattributes; then \
        echo "ERROR: faltan pesos reales y no hay .gitattributes LFS para model-folder/64x3-CNN."; \
        echo "  Sube el SavedModel a tu repo (Git LFS) y que exista en:"; \
        echo "  ${CNN_RAW_BASE}/saved_model.pb"; \
        exit 1; \
      fi; \
      echo "LFS detectado: descargando 64x3-CNN desde el proyecto (curl, como TORTUOSITY)..." && \
      BASE="${CNN_RAW_BASE}" && \
      curl -fsSL "$BASE/fingerprint.pb"    -o model-folder/64x3-CNN.model/fingerprint.pb && \
      curl -fsSL "$BASE/keras_metadata.pb" -o model-folder/64x3-CNN.model/keras_metadata.pb && \
      curl -fsSL "$BASE/saved_model.pb"    -o model-folder/64x3-CNN.model/saved_model.pb && \
      curl -fsSL "$BASE/variables/variables.index" -o model-folder/64x3-CNN.model/variables/variables.index && \
      curl -fsSL "$BASE/variables/variables.data-00000-of-00001" -o model-folder/64x3-CNN.model/variables/variables.data-00000-of-00001 && \
      echo "64x3-CNN listo."; \
    else \
      echo "64x3-CNN: pesos ya presentes en el contexto de build (checkout LFS o archivos locales)."; \
    fi

# RETFound (cuantizado): gdown en build
RUN if [ ! -f "checkpoint-quantized-model.pth" ]; then \
    echo "Downloading quantized model checkpoint from Google Drive using gdown..." && \
    ( gdown "https://drive.google.com/uc?id=1E6zIEuqAw7s6yl3st3N_jt-fNLfCegAO" -O checkpoint-quantized-model.pth \
      || gdown --fuzzy "https://drive.google.com/file/d/1E6zIEuqAw7s6yl3st3N_jt-fNLfCegAO/view?usp=sharing" -O checkpoint-quantized-model.pth ) && \
    echo "Quantized model checkpoint downloaded successfully"; \
    fi

# Create directories
RUN mkdir -p /app/diabetic_retinopathy_dataset

# Expose port
EXPOSE 8000

# Cloud Run inyecta PORT; local suele usar 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]