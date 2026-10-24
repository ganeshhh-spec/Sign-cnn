/**
 * Custom hook for translation state management.
 * Uses the backend's authoritative TextBuffer - does not duplicate buffering in frontend.
 */

import { useCallback, useEffect, useState } from 'react';
import { api } from '../api/client';
import type { StabilizedPredictionResponse } from '../types/api';

interface TranslationState {
  text: string;
  length: number;
  stabilizer: {
    window_size: number;
    last_appended: string | null;
    held_char_count: number;
    recent_predictions: string[];
  } | null;
  loading: boolean;
  error: string | null;
}

export function useTranslation() {
  const [state, setState] = useState<TranslationState>({
    text: '',
    length: 0,
    stabilizer: null,
    loading: false,
    error: null,
  });

  const [useStabilizer, setUseStabilizer] = useState(true);

  // Authoritatively sync text from backend
  const syncServerText = useCallback((text: string) => {
    setState((prev) => {
      if (prev.text === text && prev.length === text.length) return prev;
      return {
        ...prev,
        text,
        length: text.length,
        error: null,
      };
    });
  }, []);

  const fetchBuffer = useCallback(async () => {
    try {
      const response = await api.getTranslation();
      setState((prev) => ({
        ...prev,
        text: response.text,
        length: response.length,
        error: null,
      }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        error: e instanceof Error ? e.message : 'Failed to fetch translation',
      }));
    }
  }, []);

  const fetchState = useCallback(async () => {
    try {
      const response = await api.getTranslationState();
      setState((prev) => ({
        ...prev,
        text: response.buffer.text,
        length: response.buffer.length,
        stabilizer: response.stabilizer,
        error: null,
      }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        error: e instanceof Error ? e.message : 'Failed to fetch translation state',
      }));
    }
  }, []);

  const insertSpace = useCallback(async () => {
    try {
      const response = await api.insertSpace();
      setState((prev) => ({
        ...prev,
        text: response.text,
        length: response.length,
        error: null,
      }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        error: e instanceof Error ? e.message : 'Failed to insert space',
      }));
    }
  }, []);

  const deleteChar = useCallback(async () => {
    try {
      const response = await api.deleteChar();
      setState((prev) => ({
        ...prev,
        text: response.text,
        length: response.length,
        error: null,
      }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        error: e instanceof Error ? e.message : 'Failed to delete character',
      }));
    }
  }, []);

  const clear = useCallback(async () => {
    try {
      await api.clearTranslation();
      setState((prev) => ({ ...prev, text: '', length: 0, error: null }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        error: e instanceof Error ? e.message : 'Failed to clear translation',
      }));
    }
  }, []);

  const resetStabilizer = useCallback(async () => {
    try {
      await api.resetStabilizer();
      setState((prev) => ({ ...prev, stabilizer: null, error: null }));
    } catch (e) {
      setState((prev) => ({
        ...prev,
        error: e instanceof Error ? e.message : 'Failed to reset stabilizer',
      }));
    }
  }, []);

  // Handle stabilized prediction by syncing with authoritative backend buffer
  const handleStabilized = useCallback(
    async (stabilized: StabilizedPredictionResponse | null, serverText?: string | null) => {
      if (!stabilized) return;

      // If authoritative text was provided with the prediction response, update immediately
      if (serverText !== undefined && serverText !== null) {
        syncServerText(serverText);
        return;
      }

      // If an action took place on the backend without explicit serverText, fetch from server
      if (stabilized.action !== 'none') {
        await fetchBuffer();
      }
    },
    [syncServerText, fetchBuffer]
  );

  // Initial fetch
  useEffect(() => {
    fetchBuffer();
    fetchState();
  }, [fetchBuffer, fetchState]);

  return {
    text: state.text,
    length: state.length,
    stabilizer: state.stabilizer,
    loading: state.loading,
    error: state.error,
    useStabilizer,
    setUseStabilizer,
    syncServerText,
    fetchBuffer,
    fetchState,
    insertSpace,
    deleteChar,
    clear,
    resetStabilizer,
    handleStabilized,
  };
}