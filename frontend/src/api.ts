import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios'
import type { 
  AnalyzeResponse, 
  ApplySuggestionResponse, 
  ValidationResult,
  FieldSuggestion,
  StructuredCV 
} from './types';
import { sanitizeInput, detectSQLInjection } from './utils/security';

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  // Security: Set timeout to prevent hanging requests
  timeout: 30000,
  // Security: Enable credentials for CORS
  withCredentials: true,
})

API.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  
  // Security: Add request timestamp for monitoring
  config.headers['X-Request-Time'] = new Date().toISOString();
  
  return config
})

// Refresh token on 401
let isRefreshing = false
let pending: Array<(token: string) => void> = []

function onRefreshed(newToken: string) {
  pending.forEach(cb => cb(newToken))
  pending = []
}

API.interceptors.response.use(
  (res) => res,
  async (error: AxiosError) => {
    const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean }
    const status = error?.response?.status
    if (status === 401 && !original._retry) {
      original._retry = true
      const refreshToken = localStorage.getItem('refreshToken')
      if (!refreshToken) {
        // no refresh token, clear and reject
        localStorage.removeItem('token')
        return Promise.reject(error)
      }
      if (isRefreshing) {
        return new Promise(resolve => {
          pending.push((newToken: string) => {
            original.headers.Authorization = `Bearer ${newToken}`
            resolve(API(original))
          })
        })
      }
      isRefreshing = true
      try {
        const r = await API.post('/auth/refresh', { refreshToken })
        const newToken = r.data.token
        const newRefresh = r.data.refreshToken
        localStorage.setItem('token', newToken)
        if (newRefresh) localStorage.setItem('refreshToken', newRefresh)
        API.defaults.headers.common['Authorization'] = `Bearer ${newToken}`
        onRefreshed(newToken)
        return API(original)
      } catch (e) {
        localStorage.removeItem('token')
        localStorage.removeItem('refreshToken')
        return Promise.reject(e)
      } finally {
        isRefreshing = false
      }
    }
    return Promise.reject(error)
  }
)

export default API



const API_BASE_URL = import.meta.env.VITE_API_URL || (
  import.meta.env.MODE === 'development' ? 'http://127.0.0.1:8000' : ''
);

/**
 * Analyzes a CV file using structured data approach
 * @param cvFile - The CV file to analyze
 * @param jobDescription - Optional job description for targeted analysis
 * @returns Structured analysis results
 */
export async function analyzeStructuredCV(
  cvFile: File, 
  jobDescription: string = ''
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('cv_file', cvFile);
  formData.append('job_description', jobDescription);

  const response = await fetch(`${API_BASE_URL}/analyze-structured`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  if (data.error) {
    throw new Error(data.error);
  }

  if (data.gemini_analysis?.error) {
    throw new Error(data.gemini_analysis.error);
  }

  return data;
}

/**
 * Applies a suggestion to the structured CV data
 * @param structuredCV - The structured CV object
 * @param suggestion - The suggestion to apply
 * @returns Updated CV data
 */
export async function applySuggestion(
  structuredCV: StructuredCV, 
  suggestion: FieldSuggestion
): Promise<ApplySuggestionResponse> {
  const response = await fetch(`${API_BASE_URL}/apply-suggestion`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      structured_cv: structuredCV,
      suggestion: suggestion
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to apply suggestion');
  }

  return data;
}

/**
 * Validates file type and size before upload
 * @param file - File to validate
 * @returns Validation result
 */
export function validateCVFile(file: File | null): ValidationResult {
  const validTypes = ['.pdf', '.docx', '.doc', '.txt', '.odt', '.tex', '.html', '.rtf'];
  const maxSize = 10 * 1024 * 1024; // 10MB

  if (!file) {
    return { valid: false, error: 'No file provided' };
  }

  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  
  if (!validTypes.includes(ext)) {
    return { 
      valid: false, 
      error: `Unsupported file type. Please upload: ${validTypes.join(', ')}` 
    };
  }

  if (file.size > maxSize) {
    return { 
      valid: false, 
      error: 'File size exceeds 10MB limit' 
    };
  }

  return { valid: true };
}

/**
 * Export CV as LaTeX-compiled PDF
 * @param structuredCV - The structured CV object
 * @returns PDF blob for download
 */
export async function exportLatexPdf(structuredCV: StructuredCV): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/export-latex-pdf`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      structured_cv: structuredCV
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    const errorMsg = errorData?.message || errorData?.error || `HTTP error! status: ${response.status}`;
    throw new Error(errorMsg);
  }

  return await response.blob();
}

/**
 * Get CV as LaTeX code without compilation
 * @param structuredCV - The structured CV object
 * @returns LaTeX code as string
 */
export async function getLatexCode(structuredCV: StructuredCV): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/get-latex-code`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      structured_cv: structuredCV
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    const errorMsg = errorData?.message || errorData?.error || `HTTP error! status: ${response.status}`;
    throw new Error(errorMsg);
  }

  const data = await response.json();
  return data.latex_code || '';
}

