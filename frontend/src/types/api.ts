/**
 * API type definitions matching the FastAI backend contract.
 * Do not fabricate fields - these must match the actual backend responses.
 */

export interface ClassInfo {
  index: number;
  label: string;
}

export interface ClassesResponse {
  classes: ClassInfo[];
  count: number;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  model_loaded: boolean;
  device: string;
  num_classes: number;
  gpu_available: boolean;
  gpu_name: string | null;
  model_load_time_seconds: number;
  checkpoint_name?: string | null;
  checkpoint_type?: string | null;
}

export interface PredictionItem {
  rank: number;
  class_index: number;
  class_label: string;
  confidence: number;
}

export interface StabilizedPredictionResponse {
  action: 'append' | 'delete' | 'space' | 'clear' | 'none';
  character: string;
  confidence: number;
  is_stable: boolean;
  raw_prediction: string;
  raw_confidence: number;
  status_reason?: string | null;
}

export interface SampleGesture {
  class_label: string;
  filename: string;
  is_priority_demo: boolean;
  image_base64: string;
}

export interface SamplesResponse {
  samples: SampleGesture[];
  count: number;
}

export interface TrackingDiagnostics {
  landmarks_count: number;
  tracking_confidence: number;
  is_smoothed: boolean;
  crop_padding_applied: boolean;
  crop_fully_within_frame: boolean;
  crop_touches_frame_edge: boolean;
  crop_bounds: HandBBox | null;
  frame_size: { width: number; height: number } | null;
  handedness_normalized?: boolean;
  osn_applied?: boolean;
  osn_rotation_deg?: number;
  osn_hand_coverage_pct?: number;
}

export interface PredictResponse {
  success: boolean;
  predictions: PredictionItem[];
  stabilized: StabilizedPredictionResponse | null;
  hand_detected: boolean;
  hand_bbox: HandBBox | null;
  hand_handedness: string | null;
  tracking_diagnostics?: TrackingDiagnostics | null;
  inference_time_ms: number;
  preprocessing_time_ms: number;
  total_time_ms: number;
  text_buffer?: string | null;
  debug_crop_path?: string | null;
}

export interface HandBBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface TextBufferResponse {
  text: string;
  length: number;
}

export interface TranslationStateResponse {
  buffer: {
    text: string;
    length: number;
  };
  stabilizer: {
    window_size: number;
    last_appended: string | null;
    held_char_count: number;
    recent_predictions: string[];
  };
}

export interface PreprocessingConfig {
  input_size: [number, number];
  normalization: {
    mean: [number, number, number];
    std: [number, number, number];
  };
  format: 'RGB';
}

export type PredictRequest = {
  image: string; // base64 encoded JPEG/PNG
  use_stabilizer?: boolean;
  return_top_k?: number;
  capture_debug_crop?: boolean;
  enable_osn?: boolean;
};

export type ActionType = 'append' | 'delete' | 'space' | 'clear' | 'none';

export const CLASSES: readonly string[] = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  'del', 'nothing', 'space'
] as const;

export type ClassLabel = typeof CLASSES[number];

export const CLASS_TO_INDEX: Record<ClassLabel, number> = Object.fromEntries(
  CLASSES.map((c, i) => [c, i])
) as Record<ClassLabel, number>;
