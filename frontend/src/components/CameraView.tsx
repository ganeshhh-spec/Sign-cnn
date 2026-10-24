/**
 * CameraView - Live webcam preview with hand detection overlay,
 * responsive signing zone guide, and deterministic verified sample demo mode.
 */

import { useEffect, useState, useRef } from 'react';
import { useCamera, type CameraState } from '../hooks/useCamera';
import { api } from '@/api/client';
import type { PredictResponse, SampleGesture } from '@/types/api';
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
  const [trackingDiag, setTrackingDiag] = useState<PredictResponse['tracking_diagnostics']>(null);
  const [lastLatency, setLastLatency] = useState(0);

  // Signing hand preference: 'auto' | 'left' | 'right'
  // Defaults to 'auto', which adopts 'left' for left-handed signers (matching recording)
  const [signingHandPref, setSigningHandPref] = useState<'auto' | 'left' | 'right'>('auto');

  // Sample mode state
  const [sampleMode, setSampleMode] = useState(false);
  const [samples, setSamples] = useState<SampleGesture[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState(false);

  const isPredictingRef = useRef(false);
  const useStabilizerRef = useRef(useStabilizer);

  useEffect(() => {
    useStabilizerRef.current = useStabilizer;
  }, [useStabilizer]);

  // Load verified samples when entering sample mode
  useEffect(() => {
    if (sampleMode && samples.length === 0) {
      api.getSamples().then((res) => {
        setSamples(res.samples);
      }).catch((err) => {
        console.error('[CameraView] Failed to load verified samples:', err);
      });
    }
  }, [sampleMode, samples.length]);

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

  // Set up live frame callback
  useEffect(() => {
    setFrameCallback(async (frame) => {
      onFrame(frame);

      // If in sample mode, don't run camera prediction
      if (sampleMode) return;

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
  }, [onFrame, onPrediction, setFrameCallback, captureDebugCrop, sampleMode]);

  const handleStart = async () => {
    setSampleMode(false);
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

  const handleTestSample = async (sample: SampleGesture) => {
    setSelectedSample(sample.class_label);
    setLoadingSample(true);
    try {
      const response = await api.predict({
        image: sample.image_base64,
        use_stabilizer: useStabilizerRef.current,
        return_top_k: 3,
        capture_debug_crop: false,
      });
      setHandDetected(response.hand_detected);
      setHandBBox(response.hand_bbox);
      setHandHandedness(response.hand_handedness);
      setTrackingDiag(response.tracking_diagnostics ?? null);
      onPrediction(response);
      setLastLatency(response.total_time_ms);
    } catch (e) {
      console.error('[CameraView] Sample prediction failed:', e);
    } finally {
      setLoadingSample(false);
    }
  };

  // Effective signing hand: defaults to 'left' to align with user's recording, but auto-adapts when right hand is detected
  const effectiveSigningSide =
    signingHandPref !== 'auto'
      ? signingHandPref
      : handHandedness === 'Right'
      ? 'right'
      : 'left';

  const frameW = trackingDiag?.frame_size?.width ?? 640;
  const frameH = trackingDiag?.frame_size?.height ?? 480;

  // Mirrored hand bbox coordinates in %
  const handLeft = handBBox ? 100 - (handBBox.x_max / frameW) * 100 : 0;
  const handRight = handBBox ? 100 - (handBBox.x_min / frameW) * 100 : 0;
  const handTop = handBBox ? (handBBox.y_min / frameH) * 100 : 0;
  const handBottom = handBBox ? (handBBox.y_max / frameH) * 100 : 0;

  const handCenterX = (handLeft + handRight) / 2;
  const handCenterY = (handTop + handBottom) / 2;

  // Active guide zone boundaries in % (left: 6-46%, right: 54-94%, top: 14-76%)
  const zoneLeft = effectiveSigningSide === 'left' ? 6 : 54;
  const zoneRight = effectiveSigningSide === 'left' ? 46 : 94;
  const zoneTop = 14;
  const zoneBottom = 76;

  // Hand is in zone if its center is inside the active signing guide box
  const isHandInZone = Boolean(
    handDetected &&
    handBBox &&
    handCenterX >= zoneLeft - 5 &&
    handCenterX <= zoneRight + 5 &&
    handCenterY >= zoneTop - 5 &&
    handCenterY <= zoneBottom + 10
  );

  // Determine hint message based on edge proximity and tracking
  const getHintMessage = () => {
    if (sampleMode) {
      return 'SAMPLE MODE: Click a verified gesture below to test without camera noise';
    }
    if (!handDetected) {
      return `Position ${effectiveSigningSide === 'left' ? 'left' : 'right'} hand upright in the signing zone away from face`;
    }
    if (!isHandInZone) {
      return `Move hand into the ${effectiveSigningSide === 'left' ? 'left' : 'right'} signing zone (clear of face)`;
    }
    if (trackingDiag?.crop_touches_frame_edge) {
      return '⚠️ Warning: Hand near frame edge — keep centered in green box';
    }
    return 'Hand in signing zone — hold sign steady for 3 frames to append';
  };

  return (
    <div className={`camera-wrapper ${state !== 'active' && !sampleMode ? 'camera-wrapper--no-camera' : ''}`} role="region" aria-label="Camera preview">
      {/* Video element is ALWAYS mounted to preserve ref */}
      <video
        ref={videoRef}
        className="camera-video"
        autoPlay
        playsInline
        muted
        style={{ display: sampleMode ? 'none' : 'block' }}
        aria-hidden={state !== 'active' || sampleMode}
      />

      {/* Sample Mode Preview Canvas */}
      {sampleMode && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'var(--color-bg-deep)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 'var(--space-4)',
          zIndex: 3,
        }}>
          <div style={{
            fontSize: 'var(--font-size-sm)',
            fontWeight: 'var(--font-weight-semibold)',
            color: 'var(--color-accent)',
            marginBottom: 'var(--space-2)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}>
            Verified Sample Mode (Deterministic Evaluation)
          </div>
          <div style={{ fontSize: '11px', color: 'var(--color-text-muted)', marginBottom: 'var(--space-4)', textAlign: 'center', maxWidth: '440px' }}>
            Click any gesture below to feed a verified physical webcam frame directly to the CNN pipeline:
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center', maxWidth: '520px' }}>
            {samples.map((s) => (
              <button
                key={s.class_label}
                className={`btn btn--sm ${selectedSample === s.class_label ? 'btn--primary' : 'btn--ghost'}`}
                onClick={() => handleTestSample(s)}
                disabled={loadingSample}
                style={{
                  fontWeight: 'var(--font-weight-bold)',
                  minWidth: '42px',
                  border: selectedSample === s.class_label ? '1px solid var(--color-accent)' : '1px solid var(--color-border)',
                }}
              >
                {s.class_label}
                {s.is_priority_demo ? ' ★' : ''}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Placeholder overlay shown when camera is not actively streaming and not in sample mode */}
      {state !== 'active' && !sampleMode && (
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
              <p className="camera-placeholder-hint">Or switch to Sample Mode for instant pre-verified testing</p>
            </>
          )}
        </div>
      )}

      {/* Dedicated tracking coordinate layer (0 padding, full bleed, exact 1:1 match with video preview) */}
      {(state === 'active' || sampleMode) && !sampleMode && (
        <div className="camera-tracking-layer" aria-hidden="true">
          {/* Visual Signing Zone Guide aligned with intended hand position */}
          <div
            className={`signing-guide-box signing-guide-box--${effectiveSigningSide} ${
              isHandInZone ? 'signing-guide-box--active' : ''
            }`}
          >
            <span className="signing-guide-label">
              Signing Zone ({effectiveSigningSide === 'left' ? 'Left Hand' : 'Right Hand'}) • Clear of Face
            </span>
          </div>

          {/* MediaPipe Hand Bounding Box */}
          {handBBox && (
            <div
              className={`hand-bbox ${trackingDiag?.crop_padding_applied ? 'hand-bbox--padded' : ''}`}
              style={{
                left: `${handLeft}%`,
                top: `${handTop}%`,
                width: `${handRight - handLeft}%`,
                height: `${handBottom - handTop}%`,
              }}
            >
              <span className="hand-label">
                {handHandedness}
                {trackingDiag?.handedness_normalized ? ' (Norm)' : ''}
                {trackingDiag && trackingDiag.tracking_confidence > 0
                  ? ` ${Math.round(trackingDiag.tracking_confidence * 100)}%`
                  : ''}
                {isHandInZone ? ' ✓ In Zone' : ''}
                {trackingDiag?.crop_padding_applied ? ' • edge padded' : ''}
              </span>
            </div>
          )}
        </div>
      )}

      {/* UI overlay for status badges and live hints */}
      {(state === 'active' || sampleMode) && (
        <div className="camera-overlay">
          <div className="camera-status-row">
            <div className="camera-status">
              <span
                className={`status-indicator ${
                  handDetected ? 'status-indicator--detecting' : 'status-indicator--active'
                }`}
                aria-hidden="true"
              />
              <span className="status-text">
                {sampleMode
                  ? `Sample Mode${selectedSample ? ` [${selectedSample}]` : ''}`
                  : handDetected
                  ? `Hand Detected${trackingDiag?.landmarks_count ? ` (${trackingDiag.landmarks_count} pts)` : ''}${
                      trackingDiag?.is_smoothed ? ' • Smoothed' : ''
                    }`
                  : 'Camera Active'}
              </span>
            </div>

            {/* Handedness zone quick selector */}
            {!sampleMode && state === 'active' && (
              <div className="hand-selector" role="group" aria-label="Signing hand alignment">
                <button
                  type="button"
                  className={`btn btn--xs ${effectiveSigningSide === 'left' ? 'btn--primary' : 'btn--ghost'}`}
                  onClick={() => setSigningHandPref('left')}
                  title="Align signing zone for Left hand"
                >
                  Left Hand
                </button>
                <button
                  type="button"
                  className={`btn btn--xs ${effectiveSigningSide === 'right' ? 'btn--primary' : 'btn--ghost'}`}
                  onClick={() => setSigningHandPref('right')}
                  title="Align signing zone for Right hand"
                >
                  Right Hand
                </button>
              </div>
            )}
          </div>

          {/* Live Signing Hint */}
          <div className="signing-hint" role="status">
            {getHintMessage()}
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

        <button
          className={`btn ${sampleMode ? 'btn--primary' : 'btn--ghost'}`}
          onClick={() => {
            if (state === 'active') stop();
            setSampleMode((prev) => !prev);
          }}
          aria-label="Toggle sample demo mode"
        >
          {sampleMode ? 'Exit Sample Mode' : 'Sample Mode (Demo)'}
        </button>

        {devices.length > 0 && !sampleMode && (
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
