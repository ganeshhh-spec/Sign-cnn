/**
 * PredictionPanel - Displays top predictions with confidence bars
 */

import './prediction.css';

function EmptyState({ message }: { message: string }) {
  return (
    <div className="prediction-empty">
      <svg className="prediction-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <p className="prediction-empty-text">{message}</p>
    </div>
  );
}

function StabilizedBanner({ stabilized }: { stabilized: { action: string; character: string; confidence: number; is_stable: boolean } | null }) {
  if (!stabilized || (!stabilized.is_stable && stabilized.action === 'none')) return null;
  return (
    <div className={`stabilized-banner ${stabilized.action === 'delete' ? 'stabilized-banner--delete' : stabilized.action === 'space' ? 'stabilized-banner--space' : ''}`}>
      <svg className="stabilized-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        {stabilized.action === 'delete' ? (
          <path d="M6 18L18 6M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
        ) : stabilized.action === 'space' ? (
          <rect x="3" y="9" width="18" height="6" rx="2" />
        ) : (
          <path d="M20 6L9 17l-5-5" strokeLinecap="round" strokeLinejoin="round" />
        )}
      </svg>
      <div className="stabilized-content">
        <div className="stabilized-label">
          {stabilized.action === 'append' ? 'Appending' :
           stabilized.action === 'delete' ? 'Deleting' :
           stabilized.action === 'space' ? 'Adding Space' : 'Stable'}
        </div>
        <span className="stabilized-char">{stabilized.character || '—'}</span>
      </div>
      <span className="stabilized-confidence">
        {Math.round(stabilized.confidence * 100)}% stable
      </span>
    </div>
  );
}

function renderPredictionList(response: { predictions: Array<{ class_label: string; confidence: number }>; hand_detected: boolean }) {
  if (response.predictions.length > 0) {
    return response.predictions.map((pred, index) => (
      <div key={index} className={`prediction-item ${index === 0 ? 'prediction-item--top' : ''}`}>
        <span className="prediction-rank">{index + 1}</span>
        <span className="prediction-label">{pred.class_label}</span>
        <div className="prediction-confidence">
          <div className="confidence-bar">
            <div
              className="confidence-bar-fill"
              style={{ width: `${Math.round(pred.confidence * 100)}%` }}
              role="progressbar"
              aria-valuenow={Math.round(pred.confidence * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${pred.class_label} confidence`}
            />
          </div>
          <span className="confidence-value">{Math.round(pred.confidence * 100)}%</span>
        </div>
      </div>
    ));
  }
  if (response.hand_detected) {
    return <EmptyState message="Hand detected, but confidence too low to display" />;
  }
  return <EmptyState message="No hand detected" />;
}


export function PredictionPanel({ response, stabilized }: { response: { success: boolean; predictions: Array<{ class_label: string; confidence: number }>; hand_detected: boolean; total_time_ms: number } | null; stabilized: { action: string; character: string; confidence: number; is_stable: boolean } | null }) {
  if (!response || !response.success) {
    return (
      <div className="panel prediction-panel">
        <div className="prediction-header">
          <h3 className="prediction-title">Predictions</h3>
          <span className="prediction-latency">—</span>
        </div>
        <div className="prediction-empty">
          <svg className="prediction-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <p className="prediction-empty-text">Waiting for camera frame...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel prediction-panel">
      <div className="prediction-header">
        <h3 className="prediction-title">Top Predictions</h3>
        <span className="prediction-latency">{response.total_time_ms}ms</span>
      </div>

      <StabilizedBanner stabilized={stabilized} />

      <div className="prediction-list">
        {renderPredictionList(response)}
      </div>
    </div>
  );
}
