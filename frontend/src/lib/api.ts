import axios from 'axios';

// NOTA: Este archivo ahora solo maneja las llamadas al backend de ML
// La gestión de usuarios, pacientes y predicciones se ha migrado a Supabase
// Ver: src/lib/supabaseApi.ts para las nuevas funciones

// Usa VITE_API_URL si está definida; fallback a Google Cloud Run para producción
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://diabetic-retinopathy-project-488176611125.us-central1.run.app';

// Create axios instance for ML backend
export const mlApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests (token de Supabase)
mlApi.interceptors.request.use(
  async (config) => {
    // Obtener token de sesión de Supabase
    try {
      const { data: { session } } = await import('@/lib/supabase').then(m => m.supabase.auth.getSession());
      if (session?.access_token) {
        config.headers.Authorization = `Bearer ${session.access_token}`;
      }
    } catch (error) {
      console.warn('No se pudo obtener token de sesión:', error);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle auth errors
mlApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Redirigir a login si el token no es válido
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Prediction API para el backend de ML
export const predictionAPI = {
  predict: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  predictRETFound: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/predict/retfound', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  predictCNN: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/predict/cnn', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  preprocessPreview: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/preprocess/preview', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob',
    });
  },
};

// Models API para información de modelos
export const modelsAPI = {
  getInfo: () => mlApi.get('/models/info'),
};

// Exportar la instancia principal para compatibilidad
export const api = mlApi;
export default mlApi;