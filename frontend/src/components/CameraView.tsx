/**
 * CameraView - Live webcam preview with hand detection overlay
 */

import { useEffect, useState, useRef } from 'react';
import { useCamera, type CameraState } from '../hooks/useCamera';
import { api } from '@/api/client';
import type { PredictResponse } from '@/types/api';
import './camera.css';

interface CameraViewProps {
  onFrame: (frame: { imageBase64: string; timestamp: number }) => void;
  onPrediction: (response: PredictResponse) => void;
  useStabilizer: boolean;
  captureDebugCrop: boolean;
  onCaptureDebugCropChange: (enabled: boolean) => void;
}

export function CameraView({
  onFrame,
  onPrediction,
  useStabilizer,
  captureDebugCrop,
  onCaptureDebugCropChange,
}: CameraViewProps) {
  const [handDetected, setHandDetected] = useState(false);
  const [handBBox, setHandBBox] = useState<PredictResponse['hand_bbox']>(null);
  const [handHandedness, setHandHandedness] = useState<string | null>(null);
  const [lastLatency, setLastLatency] = useState(0);

  const isPredictingRef = useRef(false);
  const useStabilizerRef = useRef(useStabilizer);
  const captureDebugCropRef = useRef(captureDebugCrop);

  useEffect(() => {
    useStabilizerRef.current = useStabilizer;
  }, [useStabilizer]);

  useEffect(() => {
    captureDebugCropRef.current = captureDebugCrop;
  }, [captureDebugCrop]);

  const {
    state,
    devices,
    selectedDeviceId,
    error,
    videoRef,
    start,
    stop,
    switchCamera,
    setFrameCallback,
  } = useCamera({ targetFps: 10 });

  // Set up frame callback for real webcam inference
  useEffect(() => {
    setFrameCallback(async (frame) => {
      onFrame(frame);

      // Send to backend for prediction without concurrent piling
      if (!isPredictingRef.current) {
        isPredictingRef.current = true;
        try {
          const response = await api.predict({
            image: frame.imageBase64,
            use_stabilizer: useStabilizerRef.current,
            return_top_k: 3,
            capture_debug_crop: captureDebugCropRef.current,
          });
          setHandDetected(response.hand_detected);
          setHandBBox(response.hand_bbox);
          setHandHandedness(response.hand_handedness);
          onPrediction(response);
          setLastLatency(response.total_time_ms);
        } catch (e) {
          console.error('[CameraView] Prediction failed:', e);
        } finally {
          isPredictingRef.current = false;
        }
      }
    });
  }, [onFrame, onPrediction, setFrameCallback]);

  const handleStart = async () => {
    await start();
  };

  const handleStop = () => {
    stop();
    setHandDetected(false);
    setHandBBox(null);
    setHandHandedness(null);
  };

  const handleSwitchCamera = async (deviceId: string) => {
    await switchCamera(deviceId);
  };

  return (
    <div className="camera-wrapper" role="region" aria-label="Camera preview">
      {/* Video element is permanently mounted to guarantee ref availability */}
      <video
        ref={videoRef}
        className="camera-video"
        autoPlay
        playsInline
        muted
        aria-hidden="true"
      />

      {/* Placeholder / State overlay when camera is not actively running */}
      {state !== 'active' && (
        <div className="camera-placeholder" role="status" aria-live="polite">
          {state === 'requesting' ? (
            <>
              <svg className="camera-placeholder-icon camera-spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
              </svg>
              <p className="camera-placeholder-text">Connecting to camera...</p>
              <p className="camera-placeholder-hint">Initializing hardware stream</p>
            </>
          ) : state === 'error' ? (
            <>
              <svg className="camera-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <p className="camera-placeholder-text">Camera unavailable</p>
              <p className="camera-placeholder-hint">{error || 'Check camera permissions in browser'}</p>
            </>
          ) : (
            <>
              <svg className="camera-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
                <path d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 0115 0z" />
              </svg>
              <p className="camera-placeholder-text">Click Start Camera to begin</p>
              <p className="camera-placeholder-hint">Webcam permission will be requested for real-time gesture recognition</p>
            </>
          )}
        </div>
      )}

      {/* Active overlay: status indicator and bounding box */}
      {state === 'active' && (
        <div className="camera-overlay">
          <div className="camera-status">
            <span
              className={`status-indicator ${
                handDetected
                  ? 'status-indicator--detecting'
                  : 'status-indicator--active'
              }`}
              aria-hidden="true"
            />
            <span className="status-text">
              {handDetected ? 'Hand Detected' : 'Camera Active'}
            </span>
          </div>

          {handBBox && (
            <div
              className="hand-bbox"
              style={{
                left: `${handBBox.x_min * 100}%`,
                top: `${handBBox.y_min * 100}%`,
                width: `${(handBBox.x_max - handBBox.x_min) * 100}%`,
                height: `${(handBBox.y_max - handBBox.y_min) * 100}%`,
              }}
              aria-hidden="true"
            >
              <span className="hand-label">
                {handHandedness ? `${handHandedness} Hand` : 'Hand'} ✓
              </span>
            </div>
          )}
        </div>
      )}

      {/* Controls positioned over the bottom of the camera box */}
      <div className="camera-controls">
        {state === 'active' ? (
          <button
            type="button"
            className="btn btn--danger"
            onClick={handleStop}
            aria-label="Stop camera"
          >
            <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="6" y="4" width="12" height="16" rx="2" />
            </svg>
            Stop Camera
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--primary"
            onClick={handleStart}
            disabled={state === 'requesting'}
            aria-label="Start camera"
          >
            <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            {state === 'requesting' ? 'Starting...' : state === 'error' ? 'Retry Camera' : 'Start Camera'}
          </button>
        )}

        {devices.length > 1 && (
          <select
            className="btn btn--ghost btn--sm"
            value={selectedDeviceId || ''}
            onChange={(e) => handleSwitchCamera(e.target.value)}
            disabled={state === 'requesting'}
            aria-label="Select camera"
          >
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </select>
        )}

        {state === 'active' && (
          <span className="status-item" style={{ alignSelf: 'center' }}>
            <span
              className={`status-dot ${handDetected ? 'status-dot--healthy' : ''}`}
              aria-hidden="true"
            />
            <span className="status-text">
              {lastLatency > 0 ? `${lastLatency}ms` : '—'}
            </span>
          </span>
        )}

        <label className="status-item" style={{ alignSelf: 'center', textTransform: 'none', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={captureDebugCrop}
            onChange={(e) => onCaptureDebugCropChange(e.target.checked)}
            style={{ marginRight: '6px' }}
          />
          Capture debug crops
        </label>
      </div>
    </div>
  );
}

export type { CameraState };
