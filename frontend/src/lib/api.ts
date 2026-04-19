import axios, { isAxiosError } from 'axios';

// NOTA: Este archivo ahora solo maneja las llamadas al backend de ML
// La gestión de usuarios, pacientes y predicciones se ha migrado a Supabase
// Ver: src/lib/supabaseApi.ts para las nuevas funciones

// Usa VITE_API_URL si está definida; fallback a Google Cloud Run para producción
const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  'https://diabetic-retinopathy-project-2-488176611125.us-central1.run.app';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Mensaje legible de errores FastAPI (`detail` string o lista de validación). */
export function getMlErrorDetail(error: unknown): string {
  if (!isAxiosError(error)) {
    return error instanceof Error ? error.message : 'Request failed';
  }
  const data = error.response?.data as { detail?: string | Array<{ msg?: string }> } | undefined;
  if (data && typeof data.detail === 'string') return data.detail;
  if (data && Array.isArray(data.detail)) {
    const parts = data.detail.map((x) => x.msg).filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return error.message;
}

function isMlWarmup503(error: unknown): boolean {
  if (!isAxiosError(error) || error.response?.status !== 503) return false;
  const d = (error.response?.data as { detail?: string } | undefined)?.detail;
  if (typeof d !== 'string') return false;
  const lower = d.toLowerCase();
  return lower.includes('still loading') || lower.includes('retry shortly');
}

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
  predictRETFound: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/predict/retfound', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  /** Reintenta mientras el backend devuelve 503 por modelo en frío (carga en CPU). */
  predictRETFoundWithWarmup: async (file: File) => {
    const maxAttempts = 12;
    const delayMs = 5000;
    let lastError: unknown;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return await predictionAPI.predictRETFound(file);
      } catch (e) {
        lastError = e;
        if (!isMlWarmup503(e) || attempt === maxAttempts - 1) throw e;
        await sleep(delayMs);
      }
    }
    throw lastError;
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

  predictCNNWithWarmup: async (file: File) => {
    const maxAttempts = 12;
    const delayMs = 5000;
    let lastError: unknown;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return await predictionAPI.predictCNN(file);
      } catch (e) {
        lastError = e;
        if (!isMlWarmup503(e) || attempt === maxAttempts - 1) throw e;
        await sleep(delayMs);
      }
    }
    throw lastError;
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