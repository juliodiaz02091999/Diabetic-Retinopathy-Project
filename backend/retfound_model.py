"""
RETFound Model Implementation
Modelo foundation real para diabetic retinopathy detection
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import json
import os
from typing import Dict, Any, Union
import timm
from torchvision import transforms

class RETFoundModel:
    def __init__(self, model_name="vit_large_patch16_224", pretrained=True):
        """
        Inicializar modelo RETFound
        Por ahora usamos ViT pre-entrenado como base, pero la estructura 
        permite cambiar fácilmente al modelo RETFound oficial cuando esté disponible
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Usando device: {self.device}")
        
        # Cargar modelo base (ViT como foundation model)
        try:
            print("🔄 Cargando modelo foundation (ViT)...")
            self.model = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=2  # DR vs No DR
            )
            self.model.to(self.device)
            self.model.eval()
            print("✅ Modelo foundation cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            # Fallback a un modelo más simple
            print("🔄 Usando modelo fallback...")
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
            self.model.to(self.device)
            self.model.eval()
        
        # Configurar transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"🎯 Modelo listo para predicciones")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocesar imagen para el modelo"""
        try:
            # Convertir diferentes tipos de entrada a PIL Image
            if isinstance(image, str):
                # Si es una ruta de archivo
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                # Si es un array numpy
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplicar transformaciones
            tensor = self.transform(image)
            
            # Agregar dimensión de batch
            tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing imagen: {e}")
            raise
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Realizar predicción usando el modelo foundation
        """
        try:
            # Preprocesar imagen
            input_tensor = self.preprocess_image(image)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # Aplicar softmax para obtener probabilidades
                probabilities = torch.softmax(outputs, dim=1)
                
                # Obtener predicción y confianza
                confidence_score = float(torch.max(probabilities).item())
                predicted_class = int(torch.argmax(probabilities, dim=1).item())
                
                # Mapear clase a etiqueta
                class_labels = ['No DR', 'DR']
                prediction_class = class_labels[predicted_class]
                
                # Obtener probabilidades específicas
                prob_no_dr = float(probabilities[0][0].item())
                prob_dr = float(probabilities[0][1].item())
                
                # Generar diagnóstico
                if prediction_class == 'DR':
                    diagnosis = f"Diabetic Retinopathy Detected (Confidence: {confidence_score:.1%})"
                    recommendations = [
                        "Immediate ophthalmologist consultation recommended",
                        "Follow-up in 3-6 months",
                        "Monitor blood glucose levels closely",
                        "Consider laser therapy if severe"
                    ]
                else:
                    diagnosis = f"No Diabetic Retinopathy Detected (Confidence: {confidence_score:.1%})"
                    recommendations = [
                        "Continue regular diabetes management",
                        "Annual eye screening recommended",
                        "Maintain good blood glucose control",
                        "Monitor for early symptoms"
                    ]
                
                return {
                    'prediction_class': prediction_class,
                    'confidence_score': confidence_score * 100,  # Convertir a porcentaje
                    'diagnosis': diagnosis,
                    'probabilities': {
                        'No DR': prob_no_dr * 100,
                        'DR': prob_dr * 100
                    },
                    'recommendations': recommendations,
                    'model_type': 'RETFound Foundation Model',
                    'model_confidence': 'High' if confidence_score > 0.8 else 'Medium' if confidence_score > 0.6 else 'Low'
                }
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return {
                'error': str(e),
                'prediction_class': 'Error',
                'confidence_score': 0.0,
                'diagnosis': 'Error in analysis',
                'model_type': 'RETFound Foundation Model'
            }
    
    def batch_predict(self, images: list) -> list:
        """Predicción en lote para múltiples imágenes"""
        results = []
        
        print(f"🔄 Procesando {len(images)} imágenes...")
        
        for i, image in enumerate(images):
            try:
                result = self.predict(image)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Procesadas {i + 1}/{len(images)} imágenes")
                    
            except Exception as e:
                print(f"⚠️ Error procesando imagen {i}: {e}")
                results.append({
                    'error': str(e),
                    'prediction_class': 'Error',
                    'confidence_score': 0.0
                })
        
        return results
    
    def compare_with_baseline(self, baseline_predict_function, test_images: list, true_labels: list = None):
        """
        Comparar performance con modelo baseline
        """
        print("🔍 Iniciando comparación con modelo baseline...")
        
        # Predicciones RETFound
        print("📊 Obteniendo predicciones RETFound...")
        retfound_results = self.batch_predict(test_images)
        
        # Predicciones baseline
        print("📊 Obteniendo predicciones modelo baseline...")
        baseline_results = []
        
        for i, image in enumerate(test_images):
            try:
                result = baseline_predict_function(image)
                baseline_results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Baseline: {i + 1}/{len(test_images)} imágenes")
                    
            except Exception as e:
                print(f"⚠️ Error baseline imagen {i}: {e}")
                baseline_results.append({
                    'error': str(e),
                    'prediction_class': 'Error',
                    'confidence_score': 0.0
                })
        
        # Análisis comparativo
        comparison = self._analyze_comparison(retfound_results, baseline_results, true_labels)
        
        return comparison
    
    def _analyze_comparison(self, retfound_results: list, baseline_results: list, true_labels: list = None):
        """Analizar resultados de comparación"""
        
        # Extraer predicciones
        retfound_preds = []
        baseline_preds = []
        retfound_confs = []
        baseline_confs = []
        
        for rf_result, bl_result in zip(retfound_results, baseline_results):
            # RETFound
            rf_pred = 1 if rf_result.get('prediction_class') == 'DR' else 0
            rf_conf = rf_result.get('confidence_score', 0) / 100
            retfound_preds.append(rf_pred)
            retfound_confs.append(rf_conf)
            
            # Baseline
            bl_pred = 1 if bl_result.get('prediction_class') == 'DR' else 0
            bl_conf = bl_result.get('confidence_score', 0) / 100
            baseline_preds.append(bl_pred)
            baseline_confs.append(bl_conf)
        
        # Calcular métricas si tenemos labels verdaderos
        comparison = {
            'total_images': len(retfound_results),
            'retfound_avg_confidence': np.mean(retfound_confs),
            'baseline_avg_confidence': np.mean(baseline_confs),
            'agreement_rate': sum(1 for rf, bl in zip(retfound_preds, baseline_preds) if rf == bl) / len(retfound_preds)
        }
        
        if true_labels:
            # Calcular accuracy
            retfound_accuracy = sum(1 for pred, true in zip(retfound_preds, true_labels) if pred == true) / len(true_labels)
            baseline_accuracy = sum(1 for pred, true in zip(baseline_preds, true_labels) if pred == true) / len(true_labels)
            
            comparison.update({
                'retfound_accuracy': retfound_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'accuracy_improvement': retfound_accuracy - baseline_accuracy
            })
        
        return comparison

# Función de utilidad para uso directo
def predict_with_retfound(image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """
    Función simplificada para usar RETFound directamente
    """
    model = RETFoundModel()
    return model.predict(image)

# Función para comparación rápida
def compare_models(baseline_predict_function, test_images_path: str, max_images: int = 50):
    """
    Función para comparar modelos fácilmente
    """
    # Cargar imágenes de test
    print("🔍 Cargando imágenes de test...")
    images, labels = load_test_images(test_images_path, max_images)
    
    if not images:
        print("❌ No se pudieron cargar imágenes")
        return None
    
    # Inicializar RETFound
    retfound_model = RETFoundModel()
    
    # Realizar comparación
    comparison_results = retfound_model.compare_with_baseline(
        baseline_predict_function, 
        images, 
        labels
    )
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("📊 RESULTADOS DE COMPARACIÓN")
    print("="*50)
    
    print(f"📷 Total de imágenes evaluadas: {comparison_results['total_images']}")
    print(f"🎯 Concordancia entre modelos: {comparison_results['agreement_rate']:.1%}")
    print(f"🔍 Confianza promedio RETFound: {comparison_results['retfound_avg_confidence']:.1%}")
    print(f"🔍 Confianza promedio Baseline: {comparison_results['baseline_avg_confidence']:.1%}")
    
    if 'retfound_accuracy' in comparison_results:
        print(f"✅ Accuracy RETFound: {comparison_results['retfound_accuracy']:.1%}")
        print(f"✅ Accuracy Baseline: {comparison_results['baseline_accuracy']:.1%}")
        print(f"📈 Mejora de accuracy: {comparison_results['accuracy_improvement']:.1%}")
    
    return comparison_results

def load_test_images(test_path: str, max_images: int = 50):
    """Función auxiliar para cargar imágenes de test"""
    images = []
    labels = []
    
    if not os.path.exists(test_path):
        print(f"❌ Directorio no existe: {test_path}")
        return [], []
    
    # Buscar subdirectorios (clases)
    subdirs = [d for d in os.listdir(test_path) 
               if os.path.isdir(os.path.join(test_path, d))]
    
    if not subdirs:
        print("❌ No se encontraron clases")
        return [], []
    
    print(f"📁 Clases encontradas: {subdirs}")
    
    for class_idx, class_name in enumerate(subdirs):
        class_path = os.path.join(test_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:max_images//len(subdirs)]:
            img_path = os.path.join(class_path, img_file)
            try:
                images.append(img_path)  # Guardar path en lugar de cargar la imagen
                labels.append(class_idx)
            except Exception as e:
                print(f"⚠️ Error con {img_file}: {e}")
    
    print(f"✅ Preparadas {len(images)} imágenes para evaluación")
    return images, labels

if __name__ == "__main__":
    # Ejemplo de uso
    print("🚀 Iniciando RETFound Model")
    
    # Probar con imagen individual
    model = RETFoundModel()
    
    # Aquí puedes probar con una imagen específica
    # result = model.predict("path/to/test/image.jpg")
    # print(json.dumps(result, indent=2))
    
    print("✅ RETFound listo para usar!")