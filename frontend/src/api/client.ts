/**
 * API client for the ASL Recognition backend.
 * Uses axios with base URL configured for local development.
 */

import axios from 'axios';
import type { AxiosInstance, AxiosError } from 'axios';
import type {
  HealthResponse,
  ClassesResponse,
  PredictResponse,
  TextBufferResponse,
  TranslationStateResponse,
  PreprocessingConfig,
  PredictRequest,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        if (import.meta.env.DEV) {
          console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (import.meta.env.DEV) {
          console.error('[API Error]', error.response?.status, error.message);
        }
        return Promise.reject(this.normalizeError(error));
      }
    );
  }

  private normalizeError(error: AxiosError): Error {
    if (error.response) {
      const data = error.response.data as { detail?: string; error?: string } | undefined;
      return new Error(data?.detail || data?.error || `HTTP ${error.response.status}: ${error.message}`);
    }
    if (error.request) {
      return new Error('No response from server. Is the backend running?');
    }
    return new Error(error.message || 'Unknown API error');
  }

  async health(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/api/health');
    return response.data;
  }

  async classes(): Promise<ClassesResponse> {
    const response = await this.client.get<ClassesResponse>('/api/classes');
    return response.data;
  }

  async predict(request: PredictRequest): Promise<PredictResponse> {
    const response = await this.client.post<PredictResponse>('/api/predict', request);
    return response.data;
  }

  async getTranslation(): Promise<TextBufferResponse> {
    const response = await this.client.post<TextBufferResponse>('/api/translate');
    return response.data;
  }

  async insertSpace(): Promise<TextBufferResponse> {
    const response = await this.client.post<TextBufferResponse>('/api/translate/space');
    return response.data;
  }

  async deleteChar(): Promise<TextBufferResponse> {
    const response = await this.client.post<TextBufferResponse>('/api/translate/delete');
    return response.data;
  }

  async clearTranslation(): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/api/translate/clear');
    return response.data;
  }

  async resetStabilizer(): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post('/api/translate/reset');
    return response.data;
  }

  async getTranslationState(): Promise<TranslationStateResponse> {
    const response = await this.client.get<TranslationStateResponse>('/api/translate/state');
    return response.data;
  }

  async getPreprocessingConfig(): Promise<PreprocessingConfig> {
    const response = await this.client.get<PreprocessingConfig>('/api/preprocessing');
    return response.data;
  }
}

export const api = new ApiClient();