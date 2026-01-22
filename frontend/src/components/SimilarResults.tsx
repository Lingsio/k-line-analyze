import { useState } from 'react';
import { TrendingUp, TrendingDown, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import type { SimilarPattern, SearchResponse } from '../types';

interface SimilarResultsProps {
  results: SearchResponse | null;
  onPatternSelect?: (pattern: SimilarPattern) => void;
}

export function SimilarResults({ results, onPatternSelect }: SimilarResultsProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [sortBy, setSortBy] = useState<'similarity' | 'return'>('similarity');

  if (!results) {
    return (
      <div className="bg-surface border border-border rounded-lg p-6 text-center text-secondary">
        Search for similar patterns to see results here
      </div>
    );
  }

  const { similar_patterns: patterns, query_info: query } = results;

  const sortedPatterns = [...patterns].sort((a, b) => {
    if (sortBy === 'similarity') {
      return b.similarity_score - a.similarity_score;
    }
    return (b.subsequent_returns.t_plus_5 || 0) - (a.subsequent_returns.t_plus_5 || 0);
  });

  const formatPercent = (value: number | null) => {
    if (value === null) return 'N/A';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(2)}%`;
  };

  const getReturnColor = (value: number | null) => {
    if (value === null) return 'text-secondary';
    return value >= 0 ? 'text-bullish' : 'text-bearish';
  };

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            Similar Patterns ({patterns.length})
          </h3>
          <div className="flex items-center gap-2">
            <span className="text-sm text-secondary">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-background border border-border rounded px-2 py-1 text-sm"
            >
              <option value="similarity">Similarity</option>
              <option value="return">Return (T+5)</option>
            </select>
          </div>
        </div>
        <div className="text-sm text-secondary mt-1">
          Query: {query.symbol} ({query.start_date} - {query.end_date})
        </div>
      </div>

      {/* Results list */}
      <div className="max-h-96 overflow-y-auto">
        {sortedPatterns.map((pattern, index) => (
          <div
            key={`${pattern.symbol}-${pattern.start_date}`}
            className="border-b border-border last:border-0"
          >
            {/* Main row */}
            <div
              className="p-4 cursor-pointer hover:bg-background/50 transition-colors"
              onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-8 text-center text-secondary font-mono">
                    #{index + 1}
                  </div>
                  <div>
                    <div className="font-semibold flex items-center gap-2">
                      {pattern.symbol}
                      <span className="text-xs px-2 py-0.5 bg-border rounded text-secondary">
                        {pattern.market.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm text-secondary">
                      {pattern.start_date} - {pattern.end_date}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  {/* Similarity score */}
                  <div className="text-right">
                    <div className="text-xs text-secondary">Similarity</div>
                    <div className="font-mono text-accent">
                      {(pattern.similarity_score * 100).toFixed(1)}%
                    </div>
                  </div>

                  {/* T+5 return */}
                  <div className="text-right min-w-[80px]">
                    <div className="text-xs text-secondary">T+5</div>
                    <div className={`font-mono flex items-center justify-end gap-1 ${getReturnColor(pattern.subsequent_returns.t_plus_5)}`}>
                      {pattern.subsequent_returns.t_plus_5 !== null && (
                        pattern.subsequent_returns.t_plus_5 >= 0 ? (
                          <TrendingUp size={14} />
                        ) : (
                          <TrendingDown size={14} />
                        )
                      )}
                      {formatPercent(pattern.subsequent_returns.t_plus_5)}
                    </div>
                  </div>

                  {/* Expand icon */}
                  <div className="text-secondary">
                    {expandedIndex === index ? (
                      <ChevronUp size={20} />
                    ) : (
                      <ChevronDown size={20} />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Expanded details */}
            {expandedIndex === index && (
              <div className="px-4 pb-4 pt-0">
                <div className="bg-background rounded p-4">
                  <div className="grid grid-cols-4 gap-4 mb-4">
                    <div>
                      <div className="text-xs text-secondary mb-1">T+1</div>
                      <div className={`font-mono ${getReturnColor(pattern.subsequent_returns.t_plus_1)}`}>
                        {formatPercent(pattern.subsequent_returns.t_plus_1)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-secondary mb-1">T+5</div>
                      <div className={`font-mono ${getReturnColor(pattern.subsequent_returns.t_plus_5)}`}>
                        {formatPercent(pattern.subsequent_returns.t_plus_5)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-secondary mb-1">T+10</div>
                      <div className={`font-mono ${getReturnColor(pattern.subsequent_returns.t_plus_10)}`}>
                        {formatPercent(pattern.subsequent_returns.t_plus_10)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-secondary mb-1">T+20</div>
                      <div className={`font-mono ${getReturnColor(pattern.subsequent_returns.t_plus_20)}`}>
                        {formatPercent(pattern.subsequent_returns.t_plus_20)}
                      </div>
                    </div>
                  </div>

                  {pattern.dtw_distance !== null && (
                    <div className="text-sm text-secondary">
                      DTW Distance: {pattern.dtw_distance.toFixed(4)}
                    </div>
                  )}

                  <div className="flex gap-2 mt-4">
                    <button
                      onClick={() => onPatternSelect?.(pattern)}
                      className="flex-1 py-2 bg-accent/10 text-accent rounded hover:bg-accent/20 transition-colors text-sm"
                    >
                      Compare Charts
                    </button>
                    <a
                      href={`https://www.tradingview.com/chart/?symbol=${pattern.symbol}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 px-4 py-2 bg-border rounded hover:bg-border/80 transition-colors text-sm"
                    >
                      <ExternalLink size={14} />
                      TradingView
                    </a>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default SimilarResults;
