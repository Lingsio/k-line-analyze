// OHLCV Data Types
export interface OHLCVData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface KLineResponse {
  symbol: string;
  market: string;
  period: string;
  data: OHLCVData[];
  count: number;
}

// Search Types
export interface SearchRequest {
  symbol: string;
  market: string;
  start_date: string;
  end_date: string;
  window_size: number;
  top_k: number;
  search_scope: string[];
  include_volume: boolean;
  use_dtw_refinement: boolean;
}

export interface SubsequentReturns {
  t_plus_1: number | null;
  t_plus_5: number | null;
  t_plus_10: number | null;
  t_plus_20: number | null;
}

export interface SimilarPattern {
  symbol: string;
  market: string;
  start_date: string;
  end_date: string;
  similarity_score: number;
  dtw_distance: number | null;
  subsequent_returns: SubsequentReturns;
  kline_data?: OHLCVData[];
}

export interface AnalysisSummary {
  total_matches: number;
  win_rate_1d: number;
  win_rate_5d: number;
  win_rate_20d: number;
  avg_return_1d: number;
  avg_return_5d: number;
  avg_return_20d: number;
  median_return_5d: number;
  confidence: number;
}

export interface QueryInfo {
  symbol: string;
  market: string;
  start_date: string;
  end_date: string;
  window_size: number;
  normalized_close?: number[];
}

export interface SearchResponse {
  query_info: QueryInfo;
  similar_patterns: SimilarPattern[];
  analysis: AnalysisSummary;
}

// Market Types
export interface MarketInfo {
  market: string;
  name: string;
  description: string;
  supported: boolean;
}

// Selection Types
export interface TimeRange {
  start: string;
  end: string;
}

// Chart Types
export interface ChartSelection {
  startIndex: number;
  endIndex: number;
  startDate: string;
  endDate: string;
}
