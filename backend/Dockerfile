FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gdown

RUN pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir "opencv-python-headless==4.8.1.78"

# Código + model-folder/64x3-CNN.model como archivos normales en Git (sin LFS)
COPY . .

RUN test -f model-folder/64x3-CNN.model/saved_model.pb \
    && SIZE="$(wc -c < model-folder/64x3-CNN.model/saved_model.pb | tr -d ' ')" \
    && test "$SIZE" -gt 10000 \
    || (echo "ERROR: falta SavedModel real en model-folder/64x3-CNN.model (commitea los .pb en Git)." && exit 1)

RUN if [ ! -f "checkpoint-quantized-model.pth" ]; then \
    echo "Downloading quantized model checkpoint from Google Drive using gdown..." && \
    ( gdown "https://drive.google.com/uc?id=1E6zIEuqAw7s6yl3st3N_jt-fNLfCegAO" -O checkpoint-quantized-model.pth \
      || gdown --fuzzy "https://drive.google.com/file/d/1E6zIEuqAw7s6yl3st3N_jt-fNLfCegAO/view?usp=sharing" -O checkpoint-quantized-model.pth ) && \
    echo "Quantized model checkpoint downloaded successfully"; \
    fi

# GradeNet v4 (.keras) desde Google Drive (evita commitear archivos >100MB)
RUN if [ ! -f "backend/best_model.keras" ]; then \
    echo "Downloading GradeNet .keras from Google Drive using gdown..." && \
    mkdir -p backend && \
    ( gdown "https://drive.google.com/uc?id=1axSWwVptlXErAWxbS1uigH99XGswHDm7" -O backend/best_model.keras \
      || gdown --fuzzy "https://drive.google.com/file/d/1axSWwVptlXErAWxbS1uigH99XGswHDm7/view?usp=sharing" -O backend/best_model.keras ) && \
    echo "GradeNet .keras downloaded successfully"; \
    fi

RUN mkdir -p /app/diabetic_retinopathy_dataset

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
