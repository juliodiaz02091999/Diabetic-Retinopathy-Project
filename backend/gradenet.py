import os
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

# Nota: este archivo se usa desde la API. Evitamos ejecutar inferencias/plots al importar.


DEFAULT_MODEL_PATH = os.environ.get("GRADENET_MODEL_PATH", "backend/best_model.keras")
DEFAULT_IMG_SIZE = int(os.environ.get("GRADENET_IMG_SIZE", "380"))

# Thresholds de tu v4 reconstruido
DEFAULT_THRESHOLDS = [0.54843989, 1.54118131, 2.44684135, 3.34972389]

DEFAULT_CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# ==========================================
# PREPROCESAMIENTO
# ==========================================
def crop_retina_numpy(img_bgr: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tol

    if mask.sum() == 0:
        return img_bgr

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = img_bgr[y0:y1, x0:x1]
    if cropped.size == 0:
        return img_bgr

    return cropped

def preprocess_retina_image(img_path: str, img_size: int) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")

    # crop del fondo negro
    img_bgr = crop_retina_numpy(img_bgr, tol=7)

    # resize
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))

    # realce suave usado en entrenamiento
    img_bgr = cv2.addWeighted(
        img_bgr, 4,
        cv2.GaussianBlur(img_bgr, (0, 0), 10),
        -4, 128
    )

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # preprocess de EfficientNet
    img_rgb = tf.keras.applications.efficientnet.preprocess_input(img_rgb)

    return img_rgb

# ==========================================
# POSTPROCESO V4
# ==========================================
def expected_score(prob_vec: np.ndarray) -> float:
    classes = np.arange(len(prob_vec), dtype=np.float32)
    return float(np.sum(prob_vec * classes))

def apply_thresholds(score: float, thresholds: list[float]) -> int:
    t1, t2, t3, t4 = thresholds
    if score < t1:
        return 0
    elif score < t2:
        return 1
    elif score < t3:
        return 2
    elif score < t4:
        return 3
    else:
        return 4

class GradeNetV4:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        img_size: int = DEFAULT_IMG_SIZE,
        thresholds: list[float] | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.img_size = img_size
        self.thresholds = thresholds or list(DEFAULT_THRESHOLDS)
        self.class_names = class_names or dict(DEFAULT_CLASS_NAMES)
        self.model: tf.keras.Model | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GradeNet .keras no encontrado: {self.model_path}")

        # Compat: algunos .keras se guardaron con Keras 3 (keras.src.*) y fallan con tf.keras antiguo.
        # TF >= 2.16 usa Keras 3; aun así, hacemos fallback para entornos mixtos.
        try:
            import keras  # type: ignore

            try:
                self.model = keras.saving.load_model(  # type: ignore[attr-defined]
                    self.model_path,
                    compile=False,
                    safe_mode=False,
                )
            except TypeError:
                # Keras sin safe_mode
                self.model = keras.saving.load_model(self.model_path, compile=False)  # type: ignore[attr-defined]
        except Exception:
            # Fallback final: tf.keras
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)  # type: ignore[arg-type]
            except TypeError:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def predict(self, img_path: str) -> dict[str, Any]:
        self.load()
        assert self.model is not None

        x = preprocess_retina_image(img_path, img_size=self.img_size)

        x_orig = np.expand_dims(x, axis=0)
        x_flip = np.expand_dims(np.fliplr(x), axis=0)

        probs_orig = self.model.predict(x_orig, verbose=0)[0]
        probs_flip = self.model.predict(x_flip, verbose=0)[0]
        probs_tta = (probs_orig + probs_flip) / 2.0

        pred_argmax = int(np.argmax(probs_tta))
        confidence = float(np.max(probs_tta)) * 100.0
        score = expected_score(probs_tta)
        pred_final = apply_thresholds(score, thresholds=self.thresholds)
        final_label = self.class_names.get(pred_final, str(pred_final))

        # UI-friendly: también devolver probabilidades como dict en porcentaje
        prob_dict = {
            self.class_names.get(i, str(i)): float(p) * 100.0
            for i, p in enumerate(probs_tta.tolist())
        }

        return {
            "image_path": img_path,
            "argmax_class": pred_argmax,
            "argmax_label": self.class_names.get(pred_argmax, str(pred_argmax)),
            "final_class": pred_final,
            "final_label": final_label,
            "expected_score": float(score),
            "probabilities": prob_dict,
            "confidence_score": confidence,
            "prediction_class": final_label,
            "diagnosis": f"GradeNet v4 grade: {final_label}",
            "clinical_recommendation": "Use GradeNet grade with clinical context; confirm with ophthalmologist as needed.",
            "model_used": "GradeNet v4 (EfficientNet + TTA + thresholds)",
            "model_path": self.model_path,
            "model_loaded": True,
        }