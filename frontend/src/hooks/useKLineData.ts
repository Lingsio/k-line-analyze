import { useState, useEffect, useCallback } from 'react';
import { getKLineData, getDateRange } from '../services/api';
import type { OHLCVData, KLineResponse } from '../types';

interface UseKLineDataOptions {
  symbol: string;
  market?: string;
  period?: string;
  days?: number;
  startDate?: string;
  endDate?: string;
}

interface UseKLineDataResult {
  data: OHLCVData[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
  responseInfo: Omit<KLineResponse, 'data'> | null;
}

export function useKLineData(options: UseKLineDataOptions): UseKLineDataResult {
  const {
    symbol,
    market = 'us',
    period = 'daily',
    days = 365,
    startDate,
    endDate,
  } = options;

  const [data, setData] = useState<OHLCVData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [responseInfo, setResponseInfo] = useState<Omit<KLineResponse, 'data'> | null>(null);

  const fetchData = useCallback(async () => {
    if (!symbol) {
      setData([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const dateRange = startDate && endDate
        ? { start: startDate, end: endDate }
        : getDateRange(days);

      const response = await getKLineData(
        symbol,
        market,
        period,
        dateRange.start,
        dateRange.end
      );

      setData(response.data);
      setResponseInfo({
        symbol: response.symbol,
        market: response.market,
        period: response.period,
        count: response.count,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch data';
      setError(message);
      setData([]);
    } finally {
      setLoading(false);
    }
  }, [symbol, market, period, days, startDate, endDate]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    data,
    loading,
    error,
    refresh: fetchData,
    responseInfo,
  };
}

// Hook for managing selection state
interface Selection {
  startIndex: number;
  endIndex: number;
  startDate: string;
  endDate: string;
}

interface UseSelectionResult {
  selection: Selection | null;
  setSelection: (selection: Selection | null) => void;
  clearSelection: () => void;
  isSelecting: boolean;
  setIsSelecting: (value: boolean) => void;
}

export function useSelection(): UseSelectionResult {
  const [selection, setSelection] = useState<Selection | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);

  const clearSelection = useCallback(() => {
    setSelection(null);
    setIsSelecting(false);
  }, []);

  return {
    selection,
    setSelection,
    clearSelection,
    isSelecting,
    setIsSelecting,
  };
}

// Hook for search functionality
interface UseSearchOptions {
  onSuccess?: (response: any) => void;
  onError?: (error: Error) => void;
}

export function useSearch(options: UseSearchOptions = {}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  const search = useCallback(async (request: any) => {
    setLoading(true);
    setError(null);

    try {
      const { searchSimilarPatterns } = await import('../services/api');
      const response = await searchSimilarPatterns(request);
      setResults(response);
      options.onSuccess?.(response);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed';
      setError(message);
      options.onError?.(err instanceof Error ? err : new Error(message));
      return null;
    } finally {
      setLoading(false);
    }
  }, [options]);

  const clearResults = useCallback(() => {
    setResults(null);
    setError(null);
  }, []);

  return {
    search,
    loading,
    error,
    results,
    clearResults,
  };
}
