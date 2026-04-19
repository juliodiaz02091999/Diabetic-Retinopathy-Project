"""
Comparación: 64x3-CNN (SavedModel) vs RETFound en el mismo dataset local.
"""

import os
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score

from cnn_model import CyberAjuCNN
from retfound_official import RETFoundOfficial


def load_test_images(test_path="../diabetic_retinopathy_dataset/test", max_images=10):
    """Cargar imágenes de test para comparación"""
    images = []
    labels = []
    filenames = []

    print(f"🔍 Buscando imágenes en: {test_path}")

    if not os.path.exists(test_path):
        print(f"❌ No existe el directorio: {test_path}")
        return [], [], []

    subdirs = [d for d in os.listdir(test_path)
               if os.path.isdir(os.path.join(test_path, d))]

    if not subdirs:
        files = [f for f in os.listdir(test_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for i, filename in enumerate(files[:max_images]):
            img_path = os.path.join(test_path, filename)
            try:
                images.append(img_path)
                labels.append(0 if 'no_dr' in filename.lower() else 1)
                filenames.append(filename)
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
                continue
    else:
        print(f"📁 Encontradas clases: {subdirs}")

        for class_idx, class_name in enumerate(subdirs):
            class_path = os.path.join(test_path, class_name)
            files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            count = 0
            for filename in files:
                if count >= max_images // len(subdirs):
                    break

                img_path = os.path.join(class_path, filename)
                try:
                    images.append(img_path)
                    true_label = 1 if class_name == 'DR' else 0
                    labels.append(true_label)
                    filenames.append(f"{class_name}/{filename}")
                    count += 1
                except Exception as e:
                    print(f"❌ Error cargando {filename}: {e}")
                    continue

    print(f"✅ Cargadas {len(images)} imágenes")
    return images, labels, filenames


def compare_models():
    print("🚀 INICIANDO COMPARACIÓN: 64x3-CNN vs RETFound")
    print("=" * 50)

    test_images, true_labels, filenames = load_test_images()

    if len(test_images) == 0:
        print("❌ No se encontraron imágenes para probar")
        return

    print(f"📊 Evaluando en {len(test_images)} imágenes")

    print("\n🔧 Cargando 64x3-CNN…")
    try:
        cnn = CyberAjuCNN()
        if not cnn.model_loaded:
            raise RuntimeError("CyberAjuCNN no cargó")
        print("✅ 64x3-CNN listo")
    except Exception as e:
        print(f"❌ Error cargando 64x3-CNN: {e}")
        return

    print("\n🔧 Inicializando RETFound…")
    try:
        retfound_official = RETFoundOfficial(checkpoint_path="checkpoint-quantized-model.pth")
    except Exception as e:
        print(f"❌ Error cargando RETFound: {e}")
        return

    cnn_predictions = []
    cnn_confidences = []
    retfound_predictions = []
    retfound_confidences = []

    print("\n🔄 Procesando imágenes…")

    for i, img_path in enumerate(test_images):
        print(f"Procesando {i+1}/{len(test_images)}: {filenames[i]}")

        try:
            start_time = time.time()
            cnn_result = cnn.predict(img_path)
            cnn_time = time.time() - start_time

            cnn_class = 1 if cnn_result['prediction_class'] == 'DR' else 0
            cnn_predictions.append(cnn_class)
            cnn_confidences.append(cnn_result['confidence_score'])

            start_time = time.time()
            retfound_result = retfound_official.predict(img_path)
            retfound_time = time.time() - start_time

            retfound_class = 1 if retfound_result['prediction_class'] == 'DR' else 0
            retfound_predictions.append(retfound_class)
            retfound_confidences.append(retfound_result['confidence_score'])

            print(f"  64x3-CNN: {cnn_result['prediction_class']} ({cnn_result['confidence_score']:.1f}%) - {cnn_time:.2f}s")
            print(f"  RETFound: {retfound_result['prediction_class']} ({retfound_result['confidence_score']:.1f}%) - {retfound_time:.2f}s")

        except Exception as e:
            print(f"❌ Error procesando {filenames[i]}: {e}")
            continue

    print("\n📊 RESULTADOS DE COMPARACIÓN")
    print("=" * 50)

    if len(true_labels) == len(cnn_predictions):
        cnn_accuracy = accuracy_score(true_labels, cnn_predictions)
        cnn_avg_confidence = np.mean(cnn_confidences)
        retfound_accuracy = accuracy_score(true_labels, retfound_predictions)
        retfound_avg_confidence = np.mean(retfound_confidences)

        print("🏆 64x3-CNN:")
        print(f"   Accuracy: {cnn_accuracy:.3f} ({cnn_accuracy*100:.1f}%)")
        print(f"   Confidence promedio: {cnn_avg_confidence:.1f}%")

        print("\n🚀 RETFound:")
        print(f"   Accuracy: {retfound_accuracy:.3f} ({retfound_accuracy*100:.1f}%)")
        print(f"   Confidence promedio: {retfound_avg_confidence:.1f}%")

        winner = "RETFound" if retfound_accuracy > cnn_accuracy else "64x3-CNN"
        if retfound_accuracy == cnn_accuracy:
            winner = "Empate"
        diff = abs(retfound_accuracy - cnn_accuracy) * 100

        print(f"\n🏅 GANADOR: {winner}")
        print(f"   Diferencia: {diff:.1f} puntos porcentuales")

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_images": len(test_images),
            "cnn_model": {
                "accuracy": float(cnn_accuracy),
                "avg_confidence": float(cnn_avg_confidence),
                "predictions": cnn_predictions
            },
            "retfound_model": {
                "accuracy": float(retfound_accuracy),
                "avg_confidence": float(retfound_avg_confidence),
                "predictions": retfound_predictions
            },
            "true_labels": true_labels,
            "winner": winner,
            "difference_percentage": float(diff)
        }

        with open("model_comparison_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Resultados guardados en: model_comparison_results.json")
    else:
        print("⚠️ No se pudieron calcular métricas - revisar etiquetas")


if __name__ == "__main__":
    compare_models()
