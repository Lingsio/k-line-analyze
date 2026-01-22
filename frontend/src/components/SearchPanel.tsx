import { useState } from 'react';
import { Search, Settings, Globe } from 'lucide-react';
import type { ChartSelection, SearchRequest } from '../types';

interface SearchPanelProps {
  symbol: string;
  market: string;
  selection: ChartSelection | null;
  onSearch: (request: SearchRequest) => void;
  loading?: boolean;
}

const MARKETS = [
  { value: 'us', label: 'US Stocks' },
  { value: 'tw', label: 'Taiwan' },
  { value: 'cn', label: 'China A' },
  { value: 'hk', label: 'Hong Kong' },
  { value: 'crypto', label: 'Crypto' },
];

export function SearchPanel({
  symbol,
  market,
  selection,
  onSearch,
  loading = false,
}: SearchPanelProps) {
  const [topK, setTopK] = useState(10);
  const [searchScope, setSearchScope] = useState<string[]>(['us']);
  const [useDtw, setUseDtw] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  const handleSearch = () => {
    if (!selection) return;

    onSearch({
      symbol,
      market,
      start_date: selection.startDate,
      end_date: selection.endDate,
      window_size: selection.endIndex - selection.startIndex + 1,
      top_k: topK,
      search_scope: searchScope,
      include_volume: true,
      use_dtw_refinement: useDtw,
    });
  };

  const toggleScope = (m: string) => {
    setSearchScope((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    );
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Search size={20} />
          Search Similar Patterns
        </h3>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`p-2 rounded transition-colors ${
            showSettings ? 'bg-accent/20 text-accent' : 'hover:bg-border'
          }`}
        >
          <Settings size={18} />
        </button>
      </div>

      {/* Selection info */}
      {selection ? (
        <div className="mb-4 p-3 bg-background rounded border border-border">
          <div className="text-sm text-secondary mb-1">Selected Pattern</div>
          <div className="flex items-center gap-4">
            <span className="font-mono text-accent">
              {selection.endIndex - selection.startIndex + 1} days
            </span>
            <span className="text-muted">
              {selection.startDate} → {selection.endDate}
            </span>
          </div>
        </div>
      ) : (
        <div className="mb-4 p-3 bg-background rounded border border-border text-center text-secondary">
          Click and drag on the chart to select a pattern
        </div>
      )}

      {/* Search scope */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-2">
          <Globe size={16} className="text-secondary" />
          <span className="text-sm text-secondary">Search Scope</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {MARKETS.map((m) => (
            <button
              key={m.value}
              onClick={() => toggleScope(m.value)}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                searchScope.includes(m.value)
                  ? 'bg-accent text-background'
                  : 'bg-border text-secondary hover:bg-border/80'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced settings */}
      {showSettings && (
        <div className="mb-4 p-3 bg-background rounded border border-border">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-secondary block mb-1">
                Top K Results
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
                min={1}
                max={100}
                className="w-full bg-surface border border-border rounded px-3 py-2 text-sm"
              />
            </div>
            <div className="flex items-center">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useDtw}
                  onChange={(e) => setUseDtw(e.target.checked)}
                  className="w-4 h-4 rounded border-border"
                />
                <span className="text-sm">Use DTW Refinement</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Search button */}
      <button
        onClick={handleSearch}
        disabled={!selection || loading || searchScope.length === 0}
        className={`w-full py-3 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${
          selection && !loading && searchScope.length > 0
            ? 'bg-accent hover:bg-accent/90 text-background'
            : 'bg-border text-muted cursor-not-allowed'
        }`}
      >
        {loading ? (
          <>
            <div className="spinner" />
            Searching...
          </>
        ) : (
          <>
            <Search size={18} />
            Search Similar Patterns
          </>
        )}
      </button>
    </div>
  );
}

export default SearchPanel;
