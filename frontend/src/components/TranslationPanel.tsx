/**
 * TranslationPanel - Accumulated text with controls
 */

import { useEffect } from 'react';
import { useTranslation } from '../hooks/useTranslation';
import type { StabilizedPredictionResponse } from '../types/api';
import './translation.css';

interface TranslationPanelProps {
  stabilized: StabilizedPredictionResponse | null;
  serverText?: string | null;
  onTextChange?: (text: string) => void;
}

export function TranslationPanel({ stabilized, serverText, onTextChange }: TranslationPanelProps) {
  const {
    text,
    length,
    stabilizer,
    loading,
    error,
    insertSpace,
    deleteChar,
    clear,
    resetStabilizer,
    handleStabilized,
    syncServerText,
  } = useTranslation();

  // Keep synced with authoritative serverText from predict response
  useEffect(() => {
    if (serverText !== undefined && serverText !== null) {
      syncServerText(serverText);
    }
  }, [serverText, syncServerText]);

  // Handle stabilized action from backend
  useEffect(() => {
    if (stabilized) {
      handleStabilized(stabilized, serverText);
    }
  }, [stabilized, serverText, handleStabilized]);

  // Controls
  const handleClear = async () => {
    await clear();
    await resetStabilizer();
    if (onTextChange) onTextChange('');
  };

  const handleSpace = async () => {
    await insertSpace();
  };

  const handleDelete = async () => {
    await deleteChar();
  };

  return (
    <div className="panel translation-panel">
      <div className="translation-header">
        <h3 className="translation-title">Translation</h3>
        <span className="translation-length">{length === 1 ? '1 character' : `${length} characters`}</span>
      </div>

      <div className="translation-content">
        {text ? (
          <>
            <span>{text}</span>
            <span className="translation-cursor" aria-hidden="true" />
          </>
        ) : (
          <span className="translation-placeholder">No text yet. Sign a letter to begin.</span>
        )}
      </div>

      <div className="translation-controls">
        <button
          className="btn btn--primary translation-btn"
          onClick={handleSpace}
          disabled={loading}
          title="Insert space (or perform 'space' gesture)"
        >
          <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <rect x="3" y="11" width="18" height="2" rx="1" />
          </svg>
          Space
        </button>
        <button
          className="btn btn--ghost translation-btn"
          onClick={handleDelete}
          disabled={loading || length === 0}
          title="Delete last character (or perform 'del' gesture)"
        >
          <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M6 18L18 6M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Delete
        </button>
        <button
          className="btn btn--danger translation-btn"
          onClick={handleClear}
          disabled={loading || length === 0}
          title="Clear all accumulated text"
        >
          <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Clear
        </button>
      </div>

      {/* Stabilizer debug state */}
      {stabilizer && (
        <div className="stabilizer-state">
          <div className="stabilizer-row">
            <span className="stabilizer-label">Window</span>
            <span className="stabilizer-value">{stabilizer.window_size}</span>
          </div>
          <div className="stabilizer-row">
            <span className="stabilizer-label">Last Appended</span>
            <span className="stabilizer-value">{stabilizer.last_appended || '—'}</span>
          </div>
          <div className="stabilizer-row">
            <span className="stabilizer-label">Held Count</span>
            <span className="stabilizer-value">{stabilizer.held_char_count}</span>
          </div>
          {stabilizer.recent_predictions.length > 0 && (
            <>
              <div className="stabilizer-row">
                <span className="stabilizer-label">Recent</span>
              </div>
              <div className="stabilizer-recent">
                {stabilizer.recent_predictions.map((p, i) => (
                  <span
                    key={i}
                    className={`stabilizer-pill ${i === stabilizer.recent_predictions.length - 1 ? 'stabilizer-pill--current' : ''}`}
                  >
                    {p}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {error && (
        <div style={{
          padding: 'var(--space-3) var(--space-4)',
          background: 'var(--color-error-muted)',
          border: '1px solid var(--color-error)',
          borderRadius: 'var(--radius-md)',
          color: 'var(--color-error)',
          fontSize: 'var(--font-size-sm)',
          margin: 'var(--space-3) var(--space-4)',
        }}>
          {error}
        </div>
      )}
    </div>
  );
}