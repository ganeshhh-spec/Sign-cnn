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

export function CameraView({ onFrame, onPrediction, useStabilizer, captureDebugCrop, onCaptureDebugCropChange }: CameraViewProps) {
  const [handDetected, setHandDetected] = useState(false);
  const [handBBox, setHandBBox] = useState<PredictResponse['hand_bbox']>(null);
  const [handHandedness, setHandHandedness] = useState<string | null>(null);
  const [trackingDiag, setTrackingDiag] = useState<PredictResponse['tracking_diagnostics']>(null);
  const [lastLatency, setLastLatency] = useState(0);

  const isPredictingRef = useRef(false);
  const useStabilizerRef = useRef(useStabilizer);

  useEffect(() => {
    useStabilizerRef.current = useStabilizer;
  }, [useStabilizer]);

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

  // Set up frame callback
  useEffect(() => {
    setFrameCallback(async (frame) => {
      onFrame(frame);

      // Send to backend for prediction
      if (!isPredictingRef.current) {
        isPredictingRef.current = true;
        try {
          const response = await api.predict({
            image: frame.imageBase64,
            use_stabilizer: useStabilizerRef.current,
            return_top_k: 3,
            capture_debug_crop: captureDebugCrop,
          });
          setHandDetected(response.hand_detected);
          setHandBBox(response.hand_bbox);
          setHandHandedness(response.hand_handedness);
          setTrackingDiag(response.tracking_diagnostics ?? null);
          onPrediction(response);
          setLastLatency(response.total_time_ms);
        } catch (e) {
          console.error('[CameraView] Prediction failed:', e);
        } finally {
          isPredictingRef.current = false;
        }
      }
    });
  }, [onFrame, onPrediction, setFrameCallback, captureDebugCrop]);

  const handleStart = async () => {
    await start();
  };

  const handleStop = () => {
    stop();
    setHandDetected(false);
    setHandBBox(null);
    setHandHandedness(null);
    setTrackingDiag(null);
  };

  const handleSwitchCamera = async (deviceId: string) => {
    await switchCamera(deviceId);
  };

  return (
    <div className={`camera-wrapper ${state !== 'active' ? 'camera-wrapper--no-camera' : ''}`} role="region" aria-label="Camera preview">
      {/* Video element is ALWAYS mounted to preserve ref and avoid timing race conditions */}
      <video
        ref={videoRef}
        className="camera-video"
        autoPlay
        playsInline
        muted
        aria-hidden={state !== 'active'}
      />

      {/* Placeholder overlay shown when camera is not actively streaming */}
      {state !== 'active' && (
        <div className="camera-placeholder">
          {state === 'error' ? (
            <>
              <svg className="camera-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <p className="camera-placeholder-text">Camera unavailable</p>
              <p className="camera-placeholder-hint" style={{ color: 'var(--color-error)' }}>{error || 'Check camera permissions'}</p>
            </>
          ) : state === 'requesting' ? (
            <>
              <svg className="camera-placeholder-icon camera-spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0110 10" stroke="currentColor" strokeLinecap="round" />
              </svg>
              <p className="camera-placeholder-text">Starting Camera...</p>
              <p className="camera-placeholder-hint">Accessing device and video stream</p>
            </>
          ) : (
            <>
              <svg className="camera-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
                <path d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
              </svg>
              <p className="camera-placeholder-text">Click Start Camera to begin</p>
              <p className="camera-placeholder-hint">Camera permission will be requested</p>
            </>
          )}
        </div>
      )}

      {/* Active overlay for hand detection and status */}
      {state === 'active' && (
        <div className="camera-overlay">
          <div className="camera-status">
            <span
              className={`status-indicator ${
                handDetected ? 'status-indicator--detecting' : 'status-indicator--active'
              }`}
              aria-hidden="true"
            />
            <span className="status-text">
              {handDetected
                ? `Hand Detected${trackingDiag?.landmarks_count ? ` (${trackingDiag.landmarks_count} pts)` : ''}${
                    trackingDiag?.is_smoothed ? ' • Smoothed' : ''
                  }`
                : 'Camera Active'}
            </span>
          </div>

          {/* Visual Signing Zone Guide to encourage proper hand positioning away from face */}
          <div
            className={`signing-guide-box ${handDetected ? 'signing-guide-box--active' : ''}`}
            aria-hidden="true"
          >
            <span className="signing-guide-label">
              Signing Zone • Upright & Clear of Face
            </span>
          </div>

          {handBBox && (
            <div
              className={`hand-bbox ${trackingDiag?.crop_padding_applied ? 'hand-bbox--padded' : ''}`}
              style={{
                // Capture frames are 640x480. Percentages keep the actual CNN
                // crop aligned when the preview scales; x is mirrored with video.
                left: `${100 - (handBBox.x_max / (trackingDiag?.frame_size?.width ?? 640)) * 100}%`,
                top: `${(handBBox.y_min / (trackingDiag?.frame_size?.height ?? 480)) * 100}%`,
                width: `${((handBBox.x_max - handBBox.x_min) / (trackingDiag?.frame_size?.width ?? 640)) * 100}%`,
                height: `${((handBBox.y_max - handBBox.y_min) / (trackingDiag?.frame_size?.height ?? 480)) * 100}%`,
              }}
              aria-hidden="true"
            >
              <span className="hand-label">
                {handHandedness}
                {trackingDiag?.handedness_normalized ? ' (Norm)' : ''}
                {trackingDiag && trackingDiag.tracking_confidence > 0
                  ? ` ${Math.round(trackingDiag.tracking_confidence * 100)}%`
                  : ''}
                {handDetected ? ' ✓' : ' ...'}
                {trackingDiag?.crop_padding_applied ? ' • edge padded' : ''}
              </span>
            </div>
          )}

          {/* Live Signing Hint */}
          <div className="signing-hint" role="status">
            {handDetected
              ? 'Hand tracked — hold sign steady for 3 frames to append'
              : 'Position hand upright in the signing zone away from your face'}
          </div>
        </div>
      )}

      {/* Controls positioned over the bottom of the camera box */}
      <div className="camera-controls">
        {state === 'active' ? (
          <button
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
            className="btn btn--primary"
            onClick={handleStart}
            disabled={state === 'requesting'}
          >
            <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            {state === 'requesting' ? 'Starting...' : state === 'error' ? 'Retry Camera' : 'Start Camera'}
          </button>
        )}

        {devices.length > 0 && (
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
        {state === 'active' && (
          <label className="status-item" style={{ alignSelf: 'center', textTransform: 'none' }}>
            <input
              type="checkbox"
              checked={captureDebugCrop}
              onChange={(event) => onCaptureDebugCropChange(event.target.checked)}
            />
            Capture crops (max 12)
          </label>
        )}
      </div>
    </div>
  );
}

export type { CameraState };
