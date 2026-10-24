/**
 * Main App component
 */

import { useState, useCallback, useEffect } from 'react';
import { CameraView } from '@components/CameraView';
import { PredictionPanel } from '@components/PredictionPanel';
import { TranslationPanel } from '@components/TranslationPanel';
import { ModelInfo } from '@components/ModelInfo';
import { StabilizerToggle } from '@components/StabilizerToggle';
import { api } from '@api/client';
import type { PredictResponse, StabilizedPredictionResponse } from '@/types/api';
import './globals.css';
import './layout.css';

function App() {
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [stabilized, setStabilized] = useState<StabilizedPredictionResponse | null>(null);
  const [serverText, setServerText] = useState<string>('');
  const [useStabilizer, setUseStabilizer] = useState(true);
  const [captureDebugCrop, setCaptureDebugCrop] = useState(false);
  const [health, setHealth] = useState<{ model_loaded: boolean; device: string; gpu_available: boolean } | null>(null);

  // Fetch health on mount
  useEffect(() => {
    let mounted = true;
    api.health().then((h: { model_loaded: boolean; device: string; gpu_available: boolean }) => {
      if (mounted) setHealth(h);
    });
    return () => { mounted = false; };
  }, []);

  const handlePrediction = useCallback((res: PredictResponse) => {
    setResponse(res);
    // The API result is authoritative for this exact frame.  Clearing this
    // state prevents an old stable action being displayed after tracking loss.
    setStabilized(res.stabilized ?? null);
    if (res.text_buffer !== undefined && res.text_buffer !== null) {
      setServerText(res.text_buffer);
    }
  }, []);

  return (
    <div className="app">
      <header className="header">
        <a href="/" className="header-brand" aria-label="ASL Recognition Home">
          <svg className="brand-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
            <path d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
          </svg>
          <div>
            <span className="brand-text">ASL Recognition</span>
            <span className="brand-subtitle">Real-time Sign Language Translation</span>
          </div>
        </a>

        <div className="header-status">
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
            <span className="status-text">{health?.device?.toUpperCase() || '—'}</span>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="workspace">
          {/* Camera + Prediction */}
          <section aria-labelledby="camera-heading" className="section-style">
            <CameraView
              onFrame={() => {}}
              onPrediction={handlePrediction}
              useStabilizer={useStabilizer}
              captureDebugCrop={captureDebugCrop}
              onCaptureDebugCropChange={setCaptureDebugCrop}
            />
            <PredictionPanel
              response={response}
              stabilized={stabilized}
            />
          </section>

          {/* Stabilizer toggle */}
          <StabilizerToggle enabled={useStabilizer} onToggle={() => setUseStabilizer((prev) => !prev)} />

          {/* Sidebar */}
          <aside className="sidebar" aria-labelledby="sidebar-heading">
            <TranslationPanel
              stabilized={stabilized}
              serverText={serverText}
              onTextChange={setServerText}
            />
            <ModelInfo />
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;
