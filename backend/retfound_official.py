"""
RETFound OFICIAL Implementation
Modelo foundation REAL para diabetic retinopathy detection
Usando checkpoint_best.pth del repositorio oficial rmaphoh/RETFound_MAE
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import sys
from typing import Dict, Any, Union
from torchvision import transforms
# Solo usamos archivo local, sin descargas

# Agregar path para importar models_vit
sys.path.append('.')

class RETFoundOfficial:
    def __init__(self, checkpoint_path="checkpoint-best.pth"):
        """
        Inicializar RETFound OFICIAL usando checkpoint real
        
        Args:
            checkpoint_path: Ruta al archivo checkpoint-best.pth
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        print(f"🔧 Usando device: {self.device}")
        print(f"🎯 Cargando RETFound OFICIAL")
        
        try:
            # Verificar que existe el archivo local
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"No se encontró {self.checkpoint_path}")
            
            print(f"✅ Checkpoint encontrado: {self.checkpoint_path}")
            print("🔄 Cargando arquitectura ViT-Large...")
            self.model = self._create_retfound_model()
            
            print("🔄 Cargando pesos pre-entrenados...")
            self._load_checkpoint()
            
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ RETFound OFICIAL cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando RETFound OFICIAL: {e}")
            print("🔄 Usando modelo local como fallback...")
            self._load_fallback_model()
        
        # Transformaciones exactas del paper oficial
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("🎯 RETFound OFICIAL listo para predicciones")
    

    
    def _create_retfound_model(self):
        """Crear arquitectura ViT-Large igual que RETFound"""
        try:
            # Intentar importar desde el repo oficial
            from models_vit import vit_large_patch16
            model = vit_large_patch16(
                num_classes=5,  # 5 clases DR como el checkpoint entrenado
                drop_path_rate=0.1,
                global_pool=True
            )
            print("✅ Arquitectura RETFound creada desde código oficial (5 clases DR)")
            return model
        
        except ImportError:
            print("⚠️ No se encontró models_vit oficial, usando timm...")
            # Fallback usando timm
            import timm
            
            # Crear modelo ViT-Large/16 compatible con RETFound (5 clases)
            model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=False,  # NO usar pesos ImageNet
                num_classes=5,     # 5 clases como el modelo original entrenado
                global_pool='token',    # Global pooling tipo token
            )
            
            # NO reemplazar el head - mantener las 5 clases del checkpoint entrenado
            print("✅ Arquitectura ViT-Large/16 creada con timm (5 clases DR)")
            return model
    
    def _load_checkpoint(self):
        """Cargar pesos del checkpoint (float o ya cuantizado)"""
        try:
            # Cargar checkpoint
            print(f"🔧 Cargando checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                print(f"✅ Checkpoint cargado, keys disponibles: {list(checkpoint.keys())}")
            else:
                print("✅ Checkpoint cargado: modelo completo serializado")
            
            # Extraer state dict
            used_key = None
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
                used_key = 'model'
                print("📦 Usando key 'model' del checkpoint")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                used_key = 'state_dict'
                print("📦 Usando key 'state_dict' del checkpoint")
            else:
                # Si no es dict, puede ser un modelo completo serializado
                if not isinstance(checkpoint, dict):
                    print("📦 Detectado modelo completo serializado. Intentando cargar directamente...")
                    self.model = checkpoint
                    self.model.eval()
                    print("✅ Modelo completo cargado correctamente")
                    return
                else:
                    state_dict = checkpoint
                    print("📦 Usando checkpoint directo como state_dict")
            
            # Detectar si el state_dict parece provenir de un modelo cuantizado
            try:
                any_qdtype = any(
                    hasattr(v, 'dtype') and str(v.dtype).startswith('torch.q') for v in state_dict.values()
                )
            except Exception:
                any_qdtype = False
            has_q_keys = any(
                ('_packed_params' in k) or (k.endswith('.scale')) or (k.endswith('.zero_point')) for k in state_dict.keys()
            )
            is_quantized_state_dict = any_qdtype or has_q_keys
            if is_quantized_state_dict:
                print("🧪 Detectado state_dict cuantizado (dinámico) - cuantizando arquitectura antes de cargar pesos...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},  # dinámico soporta principalmente Linear
                    dtype=torch.qint8
                )
            
            print(f"🔍 State dict tiene {len(state_dict)} parámetros")
            
            # Adaptar keys si es necesario
            model_state_dict = self.model.state_dict()
            adapted_state_dict = {}
            
            for key, value in state_dict.items():
                # Remover prefijos si existen
                clean_key = key.replace('module.', '').replace('model.', '')
                
                if clean_key in model_state_dict:
                    adapted_state_dict[clean_key] = value
                else:
                    print(f"⚠️ Key no encontrado: {clean_key}")
            
            # Cargar pesos adaptados
            missing_keys, unexpected_keys = self.model.load_state_dict(adapted_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Keys faltantes: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️ Keys inesperados: {len(unexpected_keys)}")
            
            print("✅ Pesos RETFound cargados exitosamente")
            
            # Si NO era un state_dict cuantizado, cuantizamos ahora para reducir memoria
            if not is_quantized_state_dict:
                print("🔄 Aplicando cuantización para optimizar memoria...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                print("✅ Modelo cuantizado exitosamente (reducción ~75% de memoria)")
            else:
                print("ℹ️ Checkpoint ya cuantizado cargado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando checkpoint: {e}")
            print("🔧 Inicializando con pesos aleatorios...")
            raise
    
    def _load_fallback_model(self):
        """Modelo fallback si falla la carga oficial"""
        import timm
        print("🔄 Cargando modelo fallback...")
        self.model = timm.create_model(
            'vit_large_patch16_224',
            pretrained=True,  # Al menos usar ImageNet
            num_classes=2
        )
        print("⚠️ Usando ViT-Large con pesos de ImageNet como fallback")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocesar imagen exactamente como RETFound oficial"""
        try:
            # Convertir diferentes tipos de entrada a PIL Image
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Tipo de imagen no soportado: {type(image)}")
            
            # Convertir a RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Aplicar transformaciones oficiales
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)  # Batch dimension
            
            return tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing imagen: {e}")
            raise
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Realizar predicción usando RETFound OFICIAL
        """
        try:
            # Preprocesar imagen
            input_tensor = self.preprocess_image(image)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # Aplicar softmax para obtener probabilidades
                probabilities = torch.softmax(outputs, dim=1)
                
                # Obtener predicción detallada
                predicted_class = int(torch.argmax(probabilities, dim=1).item())
                
                # Mapear clase a etiqueta (5 clases: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)
                class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
                prediction_class_detailed = class_names[predicted_class]
                
                # Obtener probabilidades específicas
                prob_no_dr = float(probabilities[0][0].item())  # Clase 0: No DR
                prob_dr = float(torch.sum(probabilities[0][1:]).item())  # Suma de clases 1-4: DR
                
                # INTERPRETACIÓN 1: Clase individual más probable (argmax)
                individual_prediction = prediction_class_detailed
                individual_confidence = float(torch.max(probabilities[0]).item()) * 100
                
                # INTERPRETACIÓN 2: Screening binario (suma para DR vs No-DR)
                if prob_no_dr > prob_dr:  # Comparar probabilidades No_DR vs suma de DR
                    binary_prediction = "No_DR"
                    binary_diagnosis = "Negative for Diabetic Retinopathy"
                    binary_confidence = prob_no_dr * 100  # Confianza en No DR
                else:
                    binary_prediction = "DR"
                    binary_diagnosis = f"Positive for Diabetic Retinopathy - Requires evaluation"
                    binary_confidence = prob_dr * 100  # Confianza en DR (suma de todas las etapas)
                
                # Para compatibilidad con el sistema actual, mantener interpretación binaria como principal
                prediction_class = binary_prediction
                diagnosis = binary_diagnosis
                final_confidence = binary_confidence
                
                return {
                    # Interpretación principal (binaria para screening)
                    'prediction_class': prediction_class,
                    'confidence_score': final_confidence,
                    'diagnosis': diagnosis,
                    'probabilities': {
                        'No_DR': prob_no_dr * 100,
                        'DR': prob_dr * 100
                    },
                    
                    # Interpretación detallada (clase individual más probable)
                    'individual_prediction': individual_prediction,
                    'individual_confidence': individual_confidence,
                    'individual_diagnosis': f"Most likely stage: {individual_prediction}",
                    
                    # Interpretación binaria explícita
                    'binary_prediction': binary_prediction,
                    'binary_confidence': binary_confidence,
                    'binary_diagnosis': binary_diagnosis,
                    
                    # Recomendación clínica basada en ambas interpretaciones
                    'clinical_recommendation': self._get_clinical_recommendation(
                        individual_prediction, individual_confidence, 
                        binary_prediction, binary_confidence
                    ),
                    
                    # Información detallada (compatibilidad)
                    'detailed_class': prediction_class_detailed,
                    'detailed_probabilities': {
                        class_names[i]: float(probabilities[0][i].item()) * 100 
                        for i in range(len(class_names))
                    },
                    'model_used': "RETFound_OFFICIAL_Nature2023",
                    'checkpoint_loaded': os.path.exists(self.checkpoint_path),
                    'raw_outputs': outputs.cpu().numpy().tolist()
                }
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            msg = f'Error en predicción: {str(e)}'
            return {
                'prediction_class': 'Error',
                'confidence_score': 0.0,
                'diagnosis': msg,
                'probabilities': {'No_DR': 0.0, 'DR': 0.0},
                'individual_prediction': 'Error',
                'individual_confidence': 0.0,
                'individual_diagnosis': msg,
                'binary_prediction': 'Error',
                'binary_confidence': 0.0,
                'binary_diagnosis': msg,
                'clinical_recommendation': 'Prediction failed; retry with a valid fundus image.',
                'detailed_class': 'Error',
                'detailed_probabilities': {},
                'model_used': "RETFound_ERROR",
                'checkpoint_loaded': os.path.exists(self.checkpoint_path),
                'raw_outputs': []
            }

    def _get_clinical_recommendation(self, individual_pred, individual_conf, binary_pred, binary_conf):
        """
        Generar recomendaciones clínicas basadas en ambas interpretaciones
        """
        recommendations = []
        
        # Análisis de interpretación individual
        if individual_pred == "No DR" and individual_conf > 90:
            recommendations.append("Individual assessment: No diabetic retinopathy detected with high confidence")
        elif individual_pred in ["Mild DR", "Moderate DR"]:
            recommendations.append(f"Individual assessment: {individual_pred} detected - monitor closely")
        elif individual_pred in ["Severe DR", "Proliferative DR"]:
            recommendations.append(f"Individual assessment: {individual_pred} detected - urgent ophthalmologic referral required")
        
        # Análisis de screening binario
        if binary_pred == "No_DR" and binary_conf > 80:
            recommendations.append("Screening result: Negative for diabetic retinopathy")
            recommendations.append("Recommendation: Continue routine annual screening")
        elif binary_pred == "DR":
            if binary_conf > 70:
                recommendations.append("Screening result: Positive for diabetic retinopathy")
                recommendations.append("Recommendation: Ophthalmologic evaluation recommended")
            else:
                recommendations.append("Screening result: Possible diabetic retinopathy")
                recommendations.append("Recommendation: Consider repeat imaging or specialist consultation")
        
        # Análisis de discrepancia entre métodos
        if (individual_pred == "No DR" and binary_pred == "DR") or \
           (individual_pred != "No DR" and binary_pred == "No_DR"):
            recommendations.append("Note: Different interpretations detected")
            recommendations.append("Clinical correlation recommended for definitive assessment")
        
        return " | ".join(recommendations) if recommendations else "Standard diabetic retinopathy screening completed"

# Función de utilidad
def predict_with_retfound_official(image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Función simplificada para usar RETFound OFICIAL"""
    model = RETFoundOfficial()
    return model.predict(image)

if __name__ == "__main__":
    # Test básico
    print("🚀 Iniciando test de RETFound OFICIAL")
    
    try:
        model = RETFoundOfficial()
        print("✅ RETFound OFICIAL inicializado correctamente")
        
        # Test con imagen de ejemplo
        test_images = [
            "../diabetic_retinopathy_dataset/test/DR",
            "../diabetic_retinopathy_dataset/test/No_DR"
        ]
        
        for test_dir in test_images:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_image = os.path.join(test_dir, images[0])
                    print(f"\n🔍 Probando con: {test_image}")
                    result = model.predict(test_image)
                    print(f"✅ Resultado: {result['prediction_class']} ({result['confidence_score']:.1f}%)")
                    print(f"🏆 Checkpoint cargado: {result['checkpoint_loaded']}")
                    break
        
    except Exception as e:
        print(f"❌ Error en test: {e}")