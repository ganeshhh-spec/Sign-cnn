/**
 * PredictionPanel - Displays top predictions with confidence bars,
 * acceptance/rejection diagnostics, and dynamic gesture guidance.
 */

import type { StabilizedPredictionResponse } from '@/types/api';
import './prediction.css';

interface PredictionPanelProps {
  response: {
    success: boolean;
    predictions: Array<{ class_label: string; confidence: number }>;
    hand_detected: boolean;
    total_time_ms: number;
    hand_handedness?: string | null;
  } | null;
  stabilized: StabilizedPredictionResponse | null;
}

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

function StatusBanner({
  stabilized,
  topPrediction,
  handDetected,
}: {
  stabilized: StabilizedPredictionResponse | null;
  topPrediction?: { class_label: string; confidence: number };
  handDetected: boolean;
}) {
  if (!handDetected) return null;

  // 1. Accepted Banner
  if (stabilized?.is_stable && stabilized.action !== 'none') {
    return (
      <div className={`stabilized-banner ${
        stabilized.action === 'delete' ? 'stabilized-banner--delete' :
        stabilized.action === 'space' ? 'stabilized-banner--space' : ''
      }`}>
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
            {stabilized.action === 'append' ? 'ACCEPTED' :
             stabilized.action === 'delete' ? 'ACCEPTED (Delete)' :
             stabilized.action === 'space' ? 'ACCEPTED (Space)' : 'ACCEPTED'}
          </div>
          <span className="stabilized-char">{stabilized.character || '—'}</span>
        </div>
        <span className="stabilized-confidence">
          {Math.round(stabilized.confidence * 100)}% stable
        </span>
      </div>
    );
  }

  // 2. Explanatory diagnostics when not appending
  if (topPrediction) {
    const isLowConf = topPrediction.confidence < 0.70 || stabilized?.status_reason === 'low_confidence';
    const isHeld = stabilized?.status_reason === 'held_duplicate';
    const isCooldown = stabilized?.status_reason === 'cooldown_active';
    const isStabilizing = stabilized?.status_reason?.startsWith('stabilizing') ?? false;

    let message = 'Hold steady to append';
    let badgeType = 'info';

    if (isLowConf) {
      message = `REJECTED: Low Confidence (${Math.round(topPrediction.confidence * 100)}% < 70% required)`;
      badgeType = 'warning';
    } else if (isHeld) {
      message = `HELD GESTURE: '${topPrediction.class_label}' already appended. Release hand or switch sign to type again.`;
      badgeType = 'neutral';
    } else if (isCooldown) {
      message = 'COOLDOWN: Transitioning between letters...';
      badgeType = 'info';
    } else if (isStabilizing) {
      message = `STABILIZING: Matching frame (${Math.round(topPrediction.confidence * 100)}%)...`;
      badgeType = 'active';
    }

    const badgeBg = badgeType === 'warning' ? 'rgba(239, 68, 68, 0.12)' :
                    badgeType === 'neutral' ? 'rgba(59, 130, 246, 0.12)' :
                    'rgba(245, 158, 11, 0.12)';
    const badgeColor = badgeType === 'warning' ? 'var(--color-error)' :
                      badgeType === 'neutral' ? '#60a5fa' :
                      'var(--color-warning)';
    const borderColor = badgeType === 'warning' ? 'rgba(239, 68, 68, 0.3)' :
                        badgeType === 'neutral' ? 'rgba(59, 130, 246, 0.3)' :
                        'rgba(245, 158, 11, 0.3)';

    return (
      <div style={{
        margin: 'var(--space-2) 0',
        padding: '6px 12px',
        background: badgeBg,
        border: `1px solid ${borderColor}`,
        borderRadius: 'var(--radius-sm)',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '11px',
        color: badgeColor,
        fontWeight: 'var(--font-weight-medium)',
      }}>
        <span style={{ fontSize: '13px' }}>
          {badgeType === 'warning' ? '⚠️' : badgeType === 'neutral' ? 'ℹ️' : '⏱️'}
        </span>
        <span>{message}</span>
      </div>
    );
  }

  return null;
}

function DynamicSignWarning({ classLabel }: { classLabel?: string }) {
  if (classLabel !== 'J' && classLabel !== 'Z') return null;
  return (
    <div style={{
      margin: 'var(--space-2) 0',
      padding: '6px 10px',
      background: 'rgba(245, 158, 11, 0.1)',
      border: '1px solid rgba(245, 158, 11, 0.4)',
      borderRadius: 'var(--radius-sm)',
      fontSize: '11px',
      color: 'var(--color-warning)',
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
    }}>
      <span>⚠️</span>
      <span>
        <strong>Dynamic Sign ({classLabel}):</strong> In ASL, '{classLabel}' requires motion. Recognition on static 2D frames is limited.
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

export function PredictionPanel({ response, stabilized }: PredictionPanelProps) {
  if (!response || !response.success) {
    return (
      <div className="panel prediction-panel">
        <div className="prediction-header">
          <h3 className="prediction-title">Predictions</h3>
          <span className="prediction-latency">—</span>
        </div>
        <EmptyState message="Waiting for camera frame..." />
      </div>
    );
  }

  const topPred = response.predictions.length > 0 ? response.predictions[0] : undefined;

  return (
    <div className="panel prediction-panel">
      <div className="prediction-header">
        <h3 className="prediction-title">Top Predictions</h3>
        <span className="prediction-latency">{response.total_time_ms}ms</span>
      </div>

      <StatusBanner
        stabilized={stabilized}
        topPrediction={topPred}
        handDetected={response.hand_detected}
      />

      <DynamicSignWarning classLabel={topPred?.class_label} />

      <div className="prediction-list">
        {renderPredictionList(response)}
      </div>
    </div>
  );
}
