import axios from 'axios';
import type {
  KLineResponse,
  SearchRequest,
  SearchResponse,
  MarketInfo,
} from '../types';

const API_BASE = '/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Stock Data API
export async function getKLineData(
  symbol: string,
  market: string = 'us',
  period: string = 'daily',
  startDate?: string,
  endDate?: string
): Promise<KLineResponse> {
  const params = new URLSearchParams({
    market,
    period,
  });

  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);

  const response = await api.get<KLineResponse>(
    `/stock/${symbol}/kline?${params.toString()}`
  );
  return response.data;
}

export async function getStockInfo(symbol: string, market: string = 'us') {
  const response = await api.get(`/stock/${symbol}/info?market=${market}`);
  return response.data;
}

export async function getMarkets(): Promise<MarketInfo[]> {
  const response = await api.get<MarketInfo[]>('/markets');
  return response.data;
}

// Search API
export async function searchSimilarPatterns(
  request: SearchRequest
): Promise<SearchResponse> {
  const response = await api.post<SearchResponse>('/search/similar', request);
  return response.data;
}

export async function getSearchStatus() {
  const response = await api.get('/search/status');
  return response.data;
}

// Analysis API
export async function getPatternAnalysis(
  patternId: string,
  lookaheadDays: number[] = [1, 5, 10, 20]
) {
  const params = lookaheadDays.map(d => `lookahead_days=${d}`).join('&');
  const response = await api.get(`/analysis/pattern/${patternId}?${params}`);
  return response.data;
}

export async function getGlobalStatistics() {
  const response = await api.get('/analysis/statistics');
  return response.data;
}

// Utility functions
export function formatDate(date: Date): string {
  return date.toISOString().split('T')[0];
}

export function getDateRange(days: number): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - days);

  return {
    start: formatDate(start),
    end: formatDate(end),
  };
}

export default api;
