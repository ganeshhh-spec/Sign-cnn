/**
 * ModelInfo - Displays model information and metrics
 */

import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { HealthResponse, ClassesResponse } from '../types/api';
import './layout.css';

interface ModelInfoProps {
  className?: string;
}

export function ModelInfo({ className = '' }: ModelInfoProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [classes, setClasses] = useState<ClassesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function fetchInfo() {
      try {
        setLoading(true);
        const [healthRes, classesRes] = await Promise.all([
          api.health(),
          api.classes(),
        ]);
        if (mounted) {
          setHealth(healthRes);
          setClasses(classesRes);
          setLoading(false);
        }
      } catch (e) {
        if (mounted) {
          setError(e instanceof Error ? e.message : 'Failed to load model info');
          setLoading(false);
        }
      }
    }

    fetchInfo();
    return () => { mounted = false; };
  }, []);

  if (loading) {
    return (
      <div className={`panel info-panel ${className}`}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 'var(--space-8)' }}>
          <span style={{ color: 'var(--color-text-muted)' }}>Loading model info...</span>
        </div>
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className={`panel info-panel ${className}`}>
        <div style={{ color: 'var(--color-error)', padding: 'var(--space-4)' }}>
          Failed to load model info: {error}
        </div>
      </div>
    );
  }

  const gpuName = health.gpu_name || 'CPU';
  const modelLoaded = health.model_loaded ? 'Loaded' : 'Not Loaded';
  const statusColor = health.gpu_available ? 'var(--color-success)' : 'var(--color-warning)';

  return (
    <div className={`panel info-panel ${className}`}>
      <h4 className="info-title">Model Information</h4>

      <div className="info-grid">
        <div className="info-row">
          <span className="info-label">Active Model</span>
          <span className="info-value info-value--mono" style={{ fontSize: '11px' }}>
            {health.checkpoint_name || 'checkpoint_best.pth'}
          </span>
        </div>
        <div className="info-row">
          <span className="info-label">Provenance</span>
          <span className="info-value info-value--accent" style={{ fontSize: '11px' }}>
            {health.checkpoint_type || 'Original (Baseline Kaggle)'}
          </span>
        </div>
        <div className="info-row">
          <span className="info-label">Architecture</span>
          <span className="info-value info-value--mono">Custom PyTorch CNN</span>
        </div>
        <div className="info-row">
          <span className="info-label">Classes</span>
          <span className="info-value info-value--accent">{health.num_classes}</span>
        </div>
        <div className="info-row">
          <span className="info-label">Device</span>
          <span className="info-value">{health.device.toUpperCase()}</span>
        </div>
        <div className="info-row">
          <span className="info-label">GPU</span>
          <span className="info-value info-value--mono">{gpuName}</span>
        </div>
        <div className="info-row">
          <span className="info-label">Status</span>
          <span className="info-value">
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: statusColor,
                }}
                aria-hidden="true"
              />
              {modelLoaded}
            </span>
          </span>
        </div>
        <div className="info-row">
          <span className="info-label">Load Time</span>
          <span className="info-value">{health.model_load_time_seconds.toFixed(2)}s</span>
        </div>
        <div className="info-row">
          <span className="info-label">Dataset</span>
          <span className="info-value">Kaggle ASL Alphabet</span>
        </div>
        <div className="info-row">
          <span className="info-label">Test Accuracy</span>
          <span className="info-value info-value--accent">99.79%</span>
        </div>
        <div className="info-row">
          <span className="info-label">Macro F1</span>
          <span className="info-value info-value--accent">0.9979</span>
        </div>
        <div className="info-row">
          <span className="info-label">Input Size</span>
          <span className="info-value info-value--mono">128 × 128 RGB</span>
        </div>
        <div className="info-row">
          <span className="info-label">Normalization</span>
          <span className="info-value info-value--mono">ImageNet mean/std</span>
        </div>
      </div>

      <details style={{ marginTop: 'var(--space-4)' }}>
        <summary style={{
          cursor: 'pointer',
          fontFamily: 'var(--font-family)',
          fontSize: 'var(--font-size-xs)',
          fontWeight: 'var(--font-weight-semibold)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: 'var(--color-text-muted)',
          padding: 'var(--space-2) 0',
        }}>
          Class Mapping (29 classes)
        </summary>
        <div style={{ marginTop: 'var(--space-3)', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: 'var(--space-2)' }}>
          {classes?.classes.map((cls, _i) => (
            <div key={cls.label} style={{
              fontFamily: 'var(--font-family)',
              fontSize: 'var(--font-size-xs)',
              padding: 'var(--space-1) var(--space-2)',
              background: 'var(--color-bg-deep)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <span style={{ fontWeight: 'var(--font-weight-semibold)', color: 'var(--color-text-primary)' }}>
                {cls.label}
              </span>
              <span style={{ color: 'var(--color-text-muted)' }}>{cls.index}</span>
            </div>
          ))}
        </div>
      </details>

      <div style={{
        marginTop: 'var(--space-4)',
        padding: 'var(--space-3)',
        background: 'rgba(245, 158, 11, 0.08)',
        border: '1px solid rgba(245, 158, 11, 0.35)',
        borderRadius: 'var(--radius-sm)',
        fontSize: 'var(--font-size-xs)',
        lineHeight: 'var(--line-height-relaxed)',
        color: 'var(--color-text-secondary)',
      }}>
        <strong style={{ color: 'var(--color-warning)' }}>Motion Gestures (J & Z):</strong> In ASL, letters <strong>J</strong> and <strong>Z</strong> require trajectory movement in the air. The CNN performs static 2D classification on single frames; recognition of J and Z in static mode is limited.
      </div>

      <p style={{
        marginTop: 'var(--space-4)',
        padding: 'var(--space-3)',
        background: 'var(--color-bg-deep)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-sm)',
        fontSize: 'var(--font-size-xs)',
        lineHeight: 'var(--line-height-relaxed)',
        color: 'var(--color-text-muted)',
      }}>
        <strong style={{ color: 'var(--color-text-primary)' }}>Note:</strong> The accuracy metrics above are from held-out dataset evaluation.
        Real-world webcam accuracy may differ due to lighting, background, hand position, and camera quality variations.
      </p>
    </div>
  );
}