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
  if (error.code === 'ECONNABORTED') {
    return `Tiempo de espera agotado (${ML_REQUEST_TIMEOUT_MS / 1000}s). El servidor no respondió; revisa red o Cloud Run.`;
  }
  return error.message;
}

/** Reintentar: modelo en carga en esta instancia, o 503 sin cuerpo (otra instancia fría / proxy). */
function isMlWarmup503(error: unknown): boolean {
  if (!isAxiosError(error) || error.response?.status !== 503) return false;
  const data = error.response?.data;
  if (data == null || typeof data !== 'object') return true;
  const d = (data as { detail?: unknown }).detail;
  if (d === undefined || d === '') return true;
  if (typeof d !== 'string') return false;
  const lower = d.toLowerCase();
  if (lower.includes('failed to load')) return false;
  return lower.includes('still loading') || lower.includes('retry shortly');
}

// Cloud Run puede mandar cada request a otra instancia (cada una carga TF/PyTorch al arrancar).
const ML_WARMUP_MAX_ATTEMPTS = 36;
// RETFound + SavedModel en CPU pueden tardar varios minutos en frío; no cortar antes que el backend.
const ML_WARMUP_MAX_ATTEMPTS_RETFOUND = 96;
const ML_WARMUP_DELAY_MS = 5000;

// Sin timeout, una petición colgada deja la UI en "Analyzing..." para siempre.
const ML_REQUEST_TIMEOUT_MS = 120_000;

// Sin Content-Type por defecto: application/json rompe FormData; multipart manual omite boundary.
export const mlApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: ML_REQUEST_TIMEOUT_MS,
});

mlApi.interceptors.request.use(
  async (config) => {
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }
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
  (error) => Promise.reject(error)
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
    return mlApi.post('/predict/retfound', formData);
  },

  /** Reintenta ante 503 (instancia fría o modelo aún cargando en CPU). */
  predictRETFoundWithWarmup: async (file: File) => {
    let lastError: unknown;
    for (let attempt = 0; attempt < ML_WARMUP_MAX_ATTEMPTS_RETFOUND; attempt++) {
      try {
        return await predictionAPI.predictRETFound(file);
      } catch (e) {
        lastError = e;
        if (!isMlWarmup503(e) || attempt === ML_WARMUP_MAX_ATTEMPTS_RETFOUND - 1) throw e;
        await sleep(ML_WARMUP_DELAY_MS);
      }
    }
    throw lastError;
  },

  predictCNN: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/predict/cnn', formData);
  },

  predictCNNWithWarmup: async (file: File) => {
    let lastError: unknown;
    for (let attempt = 0; attempt < ML_WARMUP_MAX_ATTEMPTS; attempt++) {
      try {
        return await predictionAPI.predictCNN(file);
      } catch (e) {
        lastError = e;
        if (!isMlWarmup503(e) || attempt === ML_WARMUP_MAX_ATTEMPTS - 1) throw e;
        await sleep(ML_WARMUP_DELAY_MS);
      }
    }
    throw lastError;
  },

  predictGradeNet: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/predict/gradenet', formData);
  },

  predictGradeNetWithWarmup: async (file: File) => {
    let lastError: unknown;
    for (let attempt = 0; attempt < ML_WARMUP_MAX_ATTEMPTS; attempt++) {
      try {
        return await predictionAPI.predictGradeNet(file);
      } catch (e) {
        lastError = e;
        if (!isMlWarmup503(e) || attempt === ML_WARMUP_MAX_ATTEMPTS - 1) throw e;
        await sleep(ML_WARMUP_DELAY_MS);
      }
    }
    throw lastError;
  },

  preprocessPreview: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return mlApi.post('/preprocess/preview', formData, { responseType: 'blob' });
  },
};

// Models API para información de modelos
export const modelsAPI = {
  getInfo: () => mlApi.get('/models/info'),
};

// Exportar la instancia principal para compatibilidad
export const api = mlApi;
export default mlApi;