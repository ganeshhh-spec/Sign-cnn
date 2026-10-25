/**
 * Custom hook for webcam lifecycle management.
 * Handles camera permission, device selection, frame capture, and cleanup.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export interface FrameData {
  imageBase64: string;
  timestamp: number;
}

export type CameraState = 'idle' | 'requesting' | 'active' | 'error' | 'stopped';

interface UseCameraOptions {
  targetFps?: number;
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
}

export function useCamera({
  targetFps = 10,
  width = 640,
  height = 480,
  facingMode = 'user',
}: UseCameraOptions = {}) {
  const [state, setState] = useState<CameraState>('idle');
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameCallbackRef = useRef<(frame: FrameData) => void>(() => {});
  const intervalRef = useRef<number | null>(null);
  const frameCountRef = useRef(0);
  const stateRef = useRef<CameraState>('idle');

  // Keep stateRef in sync with state
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Enumerate available cameras
  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devList = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devList
        .filter((d) => d.kind === 'videoinput')
        .map((d, index) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${index + 1}${d.deviceId ? ` (${d.deviceId.slice(0, 8)})` : ''}`,
        }));
      console.log('[Camera] Enumerated video devices:', videoDevices);
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch (e) {
      console.error('[Camera] Device enumeration failed:', e);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  // Set frame callback
  const setFrameCallback = useCallback((callback: (frame: FrameData) => void) => {
    frameCallbackRef.current = callback;
  }, []);

  const cleanup = useCallback(() => {
    if (intervalRef.current !== null) {
      window.cancelAnimationFrame(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {
          console.warn('[Camera] Error stopping track:', e);
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    frameCountRef.current = 0;
  }, []);

  // Start camera
  const start = useCallback(async () => {
    console.log('[Camera] Start requested. Current state:', stateRef.current);
    if (stateRef.current === 'active' || stateRef.current === 'requesting') {
      console.log('[Camera] Start ignored — camera is already active or requesting');
      return;
    }

    stateRef.current = 'requesting';
    setState('requesting');
    setError(null);

    // Stop any previous stream before requesting a new one
    cleanup();

    try {
      // 1. Verify browser WebRTC support
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Webcam API is not supported in this browser context (requires HTTPS or localhost).');
      }

      // 2. Wait up to 600ms for video element ref to mount if not yet attached
      let waitMs = 0;
      while (!videoRef.current && waitMs < 600) {
        await new Promise((r) => setTimeout(r, 50));
        waitMs += 50;
      }
      if (!videoRef.current) {
        throw new Error('Video element not ready — preview ref is not mounted in DOM.');
      }

      // 3. Request user media with device constraint fallback
      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? {
              deviceId: { exact: selectedDeviceId },
              width: { ideal: width },
              height: { ideal: height },
            }
          : {
              width: { ideal: width },
              height: { ideal: height },
              facingMode,
            },
        audio: false,
      };

      console.log('[Camera] Requesting getUserMedia with constraints:', JSON.stringify(constraints));
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err: unknown) {
        if (selectedDeviceId && err instanceof DOMException && err.name === 'OverconstrainedError') {
          console.warn('[Camera] Exact deviceId constraint failed (OverconstrainedError), falling back to default constraints');
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: width }, height: { ideal: height }, facingMode },
            audio: false,
          });
        } else {
          throw err;
        }
      }

      streamRef.current = stream;
      console.log('[Camera] getUserMedia succeeded. Tracks:', stream.getVideoTracks().map((t) => t.label));

      // Refresh devices to capture real hardware labels once permission is granted
      refreshDevices();

      // 4. Assign stream to video element
      const video = videoRef.current;
      if (!video) {
        throw new Error('Video element unmounted during camera startup.');
      }

      video.muted = true;
      video.playsInline = true;
      video.setAttribute('playsinline', 'true');
      video.setAttribute('muted', 'true');
      video.srcObject = stream;

      // 5. Wait for video metadata / canplay / playing
      console.log(`[Camera] Video stream attached. Current readyState=${video.readyState}, dimensions=${video.videoWidth}x${video.videoHeight}`);

      await new Promise<void>((resolve, reject) => {
        let timeoutId: number | null = null;

        const onReady = (eventName: string) => {
          cleanupListeners();
          console.log(`[Camera] Video event '${eventName}' fired (readyState=${video.readyState}, ${video.videoWidth}x${video.videoHeight})`);
          resolve();
        };

        const onError = (e: Event) => {
          cleanupListeners();
          console.error('[Camera] Video element error event:', e);
          reject(new Error('Video element failed to load stream.'));
        };

        const cleanupListeners = () => {
          if (timeoutId !== null) clearTimeout(timeoutId);
          video.removeEventListener('loadedmetadata', handleLoadedMetadata);
          video.removeEventListener('canplay', handleCanPlay);
          video.removeEventListener('playing', handlePlaying);
          video.removeEventListener('error', onError);
        };

        const handleLoadedMetadata = () => onReady('loadedmetadata');
        const handleCanPlay = () => onReady('canplay');
        const handlePlaying = () => onReady('playing');

        // Check if metadata is already available
        if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0) {
          console.log(`[Camera] Video already has metadata (${video.videoWidth}x${video.videoHeight}), resolving immediately.`);
          resolve();
          return;
        }

        video.addEventListener('loadedmetadata', handleLoadedMetadata);
        video.addEventListener('canplay', handleCanPlay);
        video.addEventListener('playing', handlePlaying);
        video.addEventListener('error', onError);

        timeoutId = window.setTimeout(() => {
          cleanupListeners();
          if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
            console.warn('[Camera] Metadata wait timed out after 10s, but readyState >= HAVE_METADATA. Proceeding.');
            resolve();
          } else {
            reject(new Error(`Video metadata load timeout after 10s (readyState=${video.readyState})`));
          }
        }, 10000);
      });

      // 6. Start playback
      console.log('[Camera] Calling video.play()...');
      try {
        await video.play();
        console.log(`[Camera] video.play() resolved successfully. Dimensions: ${video.videoWidth}x${video.videoHeight}`);
      } catch (playErr) {
        console.error('[Camera] video.play() failed:', playErr);
        throw new Error(`Video playback failed: ${playErr instanceof Error ? playErr.message : String(playErr)}`);
      }

      // 7. Verify non-zero videoWidth and videoHeight
      let dimWait = 0;
      while ((video.videoWidth === 0 || video.videoHeight === 0) && dimWait < 2000) {
        await new Promise((r) => setTimeout(r, 50));
        dimWait += 50;
      }
      console.log(`[Camera] Dimensions verified: ${video.videoWidth}x${video.videoHeight} (waited ${dimWait}ms)`);

      // 8. Prepare capture canvas
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
      }
      canvasRef.current.width = width;
      canvasRef.current.height = height;

      // 9. Start frame capture loop
      const frameInterval = 1000 / targetFps;
      let lastFrameTime = 0;

      const captureFrame = () => {
        if (stateRef.current !== 'active') return;

        const now = performance.now();
        if (now - lastFrameTime >= frameInterval) {
          lastFrameTime = now;
          if (videoRef.current && canvasRef.current && streamRef.current) {
            const v = videoRef.current;
            if (v.videoWidth > 0 && v.videoHeight > 0) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx) {
                ctx.drawImage(v, 0, 0, width, height);
                const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.8);
                const base64 = dataUrl.split(',')[1];
                frameCallbackRef.current({
                  imageBase64: base64,
                  timestamp: Date.now(),
                });
                frameCountRef.current += 1;
              }
            }
          }
        }

        if (stateRef.current === 'active') {
          intervalRef.current = window.requestAnimationFrame(captureFrame);
        }
      };

      stateRef.current = 'active';
      setState('active');
      intervalRef.current = window.requestAnimationFrame(captureFrame);
      console.log('[Camera] Camera active and frame capture loop running successfully.');
    } catch (e: unknown) {
      console.error('[Camera] Camera startup failed:', e);
      let userMessage = 'Failed to start camera';

      if (e instanceof DOMException) {
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
          userMessage = 'Camera permission denied. Please allow camera access in your browser address bar.';
        } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
          userMessage = 'No camera device found. Please connect or enable your webcam.';
        } else if (e.name === 'NotReadableError' || e.name === 'TrackStartError') {
          userMessage = 'Camera is already in use by another application (e.g. Teams, Zoom, or another tab) or blocked by Windows.';
        } else if (e.name === 'OverconstrainedError') {
          userMessage = 'Requested camera resolution or device constraints not supported by hardware.';
        } else {
          userMessage = `Camera error: ${e.message} (${e.name})`;
        }
      } else if (e instanceof Error) {
        userMessage = e.message;
      }

      setError(userMessage);
      stateRef.current = 'error';
      setState('error');
      cleanup();
    }
  }, [selectedDeviceId, width, height, facingMode, targetFps, cleanup, refreshDevices]);

  // Stop camera
  const stop = useCallback(() => {
    console.log('[Camera] Stopping camera stream...');
    cleanup();
    stateRef.current = 'stopped';
    setState('stopped');
  }, [cleanup]);

  // Switch camera
  const switchCamera = useCallback(async (deviceId: string) => {
    console.log('[Camera] Switching camera to device:', deviceId);
    const wasActive = stateRef.current === 'active';
    if (wasActive) stop();
    setSelectedDeviceId(deviceId);
    if (wasActive) await start();
  }, [start, stop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  return {
    state,
    devices,
    selectedDeviceId,
    error,
    videoRef,
    canvasRef,
    frameCountRef,
    start,
    stop,
    switchCamera,
    setFrameCallback,
    refreshDevices,
  };
}

export default useCamera;