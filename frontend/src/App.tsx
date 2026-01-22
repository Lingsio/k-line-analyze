import { useState, useCallback } from 'react';
import { BarChart3, RefreshCw, Info, Zap } from 'lucide-react';
import { KLineChart } from './components/KLineChart';
import { SearchPanel } from './components/SearchPanel';
import { SimilarResults } from './components/SimilarResults';
import { AnalysisPanel } from './components/AnalysisPanel';
import { SymbolSearch } from './components/SymbolSearch';
import { AutoAnalysis } from './components/AutoAnalysis';
import { useKLineData, useSearch } from './hooks/useKLineData';
import type { ChartSelection, SearchRequest, SearchResponse, SimilarPattern } from './types';

type TabType = 'search' | 'auto';

function App() {
  // State
  const [symbol, setSymbol] = useState('AAPL');
  const [market, setMarket] = useState('us');
  const [selection, setSelection] = useState<ChartSelection | null>(null);
  const [selectedPattern, setSelectedPattern] = useState<SimilarPattern | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('auto');

  // Data hooks
  const { data, loading: dataLoading, error, refresh, responseInfo } = useKLineData({
    symbol,
    market,
    days: 365,
  });

  const {
    search,
    loading: searchLoading,
    results,
    clearResults,
  } = useSearch();

  // Handlers
  const handleSymbolChange = useCallback((newSymbol: string) => {
    setSymbol(newSymbol);
    setSelection(null);
    clearResults();
  }, [clearResults]);

  const handleMarketChange = useCallback((newMarket: string) => {
    setMarket(newMarket);
    setSelection(null);
    clearResults();
  }, [clearResults]);

  const handleSearch = useCallback(async (request: SearchRequest) => {
    await search(request);
  }, [search]);

  const handlePatternSelect = useCallback((pattern: SimilarPattern) => {
    setSelectedPattern(pattern);
    // In a full implementation, this would load the pattern's chart data
    // for overlay comparison
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 size={28} className="text-accent" />
              <div>
                <h1 className="text-xl font-bold">K-Line Pattern Finder</h1>
                <p className="text-xs text-secondary">
                  AI-Powered Pattern Similarity Search
                </p>
              </div>
            </div>
            <SymbolSearch
              value={symbol}
              market={market}
              onSymbolChange={handleSymbolChange}
              onMarketChange={handleMarketChange}
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chart Section */}
          <div className="lg:col-span-2">
            {/* Chart info bar */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold">
                  {responseInfo?.symbol || symbol}
                </h2>
                <p className="text-sm text-secondary">
                  {responseInfo?.market?.toUpperCase()} • {responseInfo?.count || 0} data points
                </p>
              </div>
              <button
                onClick={refresh}
                disabled={dataLoading}
                className="flex items-center gap-2 px-3 py-2 bg-surface border border-border rounded hover:bg-border transition-colors"
              >
                <RefreshCw size={16} className={dataLoading ? 'animate-spin' : ''} />
                Refresh
              </button>
            </div>

            {/* Error display */}
            {error && (
              <div className="mb-4 p-4 bg-bearish/10 border border-bearish rounded-lg text-bearish">
                {error}
              </div>
            )}

            {/* Chart */}
            <div className="bg-surface border border-border rounded-lg p-4 mb-6">
              {dataLoading ? (
                <div className="h-96 flex items-center justify-center">
                  <div className="spinner" />
                </div>
              ) : data.length > 0 ? (
                <KLineChart
                  data={data}
                  selection={selection}
                  onSelectionChange={setSelection}
                  height={400}
                  showVolume={true}
                />
              ) : (
                <div className="h-96 flex items-center justify-center text-secondary">
                  No data available
                </div>
              )}
            </div>

            {/* Instructions */}
            {!selection && data.length > 0 && (
              <div className="flex items-center gap-2 p-4 bg-surface border border-border rounded-lg text-secondary">
                <Info size={18} />
                <span>
                  Click on the chart to select a start point, then drag to select
                  a pattern for similarity search.
                </span>
              </div>
            )}

            {/* Search Panel */}
            <SearchPanel
              symbol={symbol}
              market={market}
              selection={selection}
              onSearch={handleSearch}
              loading={searchLoading}
            />
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Tab Switcher */}
            <div className="flex bg-surface border border-border rounded-lg p-1">
              <button
                onClick={() => setActiveTab('auto')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
                  activeTab === 'auto'
                    ? 'bg-accent text-background'
                    : 'text-secondary hover:text-primary'
                }`}
              >
                <Zap size={16} />
                自动分析
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
                  activeTab === 'search'
                    ? 'bg-accent text-background'
                    : 'text-secondary hover:text-primary'
                }`}
              >
                <BarChart3 size={16} />
                手动搜索
              </button>
            </div>

            {/* Auto Analysis Tab */}
            {activeTab === 'auto' && (
              <AutoAnalysis symbol={symbol} market={market} />
            )}

            {/* Manual Search Tab */}
            {activeTab === 'search' && (
              <>
                {/* Similar Results */}
                <SimilarResults
                  results={results as SearchResponse}
                  onPatternSelect={handlePatternSelect}
                />

                {/* Analysis Panel */}
                <AnalysisPanel results={results as SearchResponse} />
              </>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between text-sm text-secondary">
            <div>
              K-Line Pattern Finder v1.0.0
            </div>
            <div className="flex items-center gap-4">
              <span>Data Sources: Yahoo Finance, AKShare, CCXT</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
