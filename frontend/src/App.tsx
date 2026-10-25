/**
 * Main App component
 * Integrated with KineticGrid (ambient background), GlowCard (spotlight effects),
 * and ASLBentoDashboard (Bento-style 8-tile computer-vision workspace).
 */

import { useState, useCallback, useEffect } from 'react';
import { ASLBentoDashboard } from '@/components/ui/asl-bento';
import { KineticGrid } from '@/components/ui/kinetic-grid';
import { api } from '@api/client';
import type { PredictResponse, StabilizedPredictionResponse, HealthResponse } from '@/types/api';
import './globals.css';
import './layout.css';

export function App() {
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [stabilized, setStabilized] = useState<StabilizedPredictionResponse | null>(null);
  const [serverText, setServerText] = useState<string>('');
  const [useStabilizer, setUseStabilizer] = useState(true);
  const [captureDebugCrop, setCaptureDebugCrop] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [bgAnimation, setBgAnimation] = useState(true);

  // Fetch health on mount
  useEffect(() => {
    let mounted = true;
    api.health()
      .then((h) => {
        if (mounted) setHealth(h);
      })
      .catch((err) => {
        console.warn('[App] Health check failed on mount:', err);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handlePrediction = useCallback((res: PredictResponse) => {
    setResponse(res);
    setStabilized(res.stabilized ?? null);
    if (res.text_buffer !== undefined && res.text_buffer !== null) {
      setServerText(res.text_buffer);
    }
  }, []);

  return (
    <KineticGrid disabled={!bgAnimation} globalColor="default">
      <div className="app">
        {/* Header */}
        <header className="header">
          <a href="/" className="header-brand" aria-label="ASL Recognition Home">
            <svg
              className="brand-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              aria-hidden="true"
            >
              <path d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
              <path d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
            </svg>
            <div className="brand-title-group">
              <span className="brand-text">ASL Recognition</span>
              <span className="brand-subtitle">Real-time Sign Language Translation</span>
            </div>
          </a>

          <div className="header-status">
            {/* Grid FX Toggle */}
            <button
              type="button"
              onClick={() => setBgAnimation((prev) => !prev)}
              className="btn btn--ghost btn--xs"
              style={{ fontSize: '11px', padding: '4px 8px' }}
              title="Toggle kinetic background animation"
              aria-label="Toggle background animation"
            >
              <span
                className={`status-dot ${bgAnimation ? 'status-dot--healthy' : ''}`}
                aria-hidden="true"
              />
              Grid FX: {bgAnimation ? 'On' : 'Off'}
            </button>

            {/* Health and Hardware status items */}
            <div className="status-item">
              <span
                className={`status-dot ${health?.model_loaded ? 'status-dot--healthy' : 'status-dot--degraded'}`}
                aria-hidden="true"
              />
              <span className="status-text">{health?.model_loaded ? 'Model Ready' : 'Loading...'}</span>
            </div>
            <div className="status-item">
              <span
                className={`status-dot ${health?.gpu_available ? 'status-dot--healthy' : 'status-dot--degraded'}`}
                aria-hidden="true"
              />
              <span className="status-text">{health?.device ? health.device.toUpperCase() : '—'}</span>
            </div>
          </div>
        </header>

        {/* Bento Workspace */}
        <main className="main-bento">
          <ASLBentoDashboard
            response={response}
            stabilized={stabilized}
            serverText={serverText}
            useStabilizer={useStabilizer}
            captureDebugCrop={captureDebugCrop}
            health={health}
            onPrediction={handlePrediction}
            onTextChange={setServerText}
            onToggleStabilizer={() => setUseStabilizer((prev) => !prev)}
            onCaptureDebugCropChange={setCaptureDebugCrop}
          />
        </main>
      </div>
    </KineticGrid>
  );
}

export default App;
