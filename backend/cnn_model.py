import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "model-folder/64x3-CNN.model"

# Class labels: index 0 = DR, index 1 = No DR
CLASS_LABELS = ["Diabetic Retinopathy Detected", "No Diabetic Retinopathy Detected"]
BINARY_LABELS = ["DR", "NO-DR"]

CLINICAL_RECOMMENDATIONS = {
    "NO-DR": "No signs of diabetic retinopathy detected. Continue regular annual screenings and maintain good blood sugar control.",
    "DR": "Signs of diabetic retinopathy detected. Please consult an ophthalmologist promptly for a comprehensive dilated eye exam and further evaluation.",
}


class CyberAjuCNN:
    """
    Wrapper for the 64x3-CNN SavedModel from Cyber-Aju/Diabetic_retinopathy.
    Input:  224x224 RGB image, pixel values in [0, 1]
    Output: 2-class softmax — index 0 = DR, index 1 = No DR
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._infer = None
        self.model_loaded = False
        self._load()

    def _load(self):
        try:
            loaded = tf.saved_model.load(self.model_path)
            self._infer = loaded.signatures["serving_default"]
            self.model_loaded = True
            print(f"✅ 64x3-CNN model loaded from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading 64x3-CNN model: {e}")

    def _preprocess(self, image_path: str) -> tf.Tensor:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX=10), -4, 128)
        arr = np.array(img, dtype=np.float32) / 255.0
        return tf.constant([arr], dtype=tf.float32)

    def predict(self, image_path: str) -> dict:
        if not self.model_loaded or self._infer is None:
            raise RuntimeError("64x3-CNN model is not loaded")

        tensor = self._preprocess(image_path)
        output = self._infer(tensor)

        # The model exposes its output under the key 'dense_1'
        probabilities: list[float] = output["dense_1"].numpy()[0].tolist()

        predicted_index = int(np.argmax(probabilities))
        prediction_class = BINARY_LABELS[predicted_index]
        diagnosis = CLASS_LABELS[predicted_index]
        confidence_score = float(probabilities[predicted_index]) * 100

        return {
            "confidence_score": round(confidence_score, 2),
            "prediction_class": prediction_class,
            "diagnosis": diagnosis,
            "probabilities": {
                "DR": round(probabilities[0] * 100, 2),
                "NO-DR": round(probabilities[1] * 100, 2),
            },
            "clinical_recommendation": CLINICAL_RECOMMENDATIONS[prediction_class],
            "model_used": "64x3-CNN (Cyber-Aju)",
            "model_loaded": self.model_loaded,
        }
