import { useState, useCallback } from 'react';
import { BarChart3, RefreshCw, Info, Zap, X, ArrowLeft, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import { KLineChart } from './components/KLineChart';
import { SearchPanel } from './components/SearchPanel';
import { SimilarResults } from './components/SimilarResults';
import { AnalysisPanel } from './components/AnalysisPanel';
import { SymbolSearch } from './components/SymbolSearch';
import { AutoAnalysis } from './components/AutoAnalysis';
import { useKLineData, useSearch } from './hooks/useKLineData';
import type { ChartSelection, SearchRequest, SearchResponse, SimilarPattern } from './types';

type TabType = 'search' | 'auto';

// Pattern from AutoAnalysis has different type structure
interface AutoPattern {
  symbol: string;
  market: string;
  start_date: string;  // Historical pattern's start date
  end_date: string;    // Historical pattern's end date
  current_start_date?: string;  // Current stock's pattern start date
  current_end_date?: string;    // Current stock's pattern end date
  similarity_score: number;
  subsequent_returns: {
    "t+1": number;
    "t+5": number;
    "t+10": number;
    "t+20": number;
  };
}

function App() {
  // State
  const [symbol, setSymbol] = useState('AAPL');
  const [market, setMarket] = useState('us');
  const [selection, setSelection] = useState<ChartSelection | null>(null);
  const [selectedPattern, setSelectedPattern] = useState<SimilarPattern | null>(null);
  const [autoSelectedPattern, setAutoSelectedPattern] = useState<AutoPattern | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('auto');
  const [comparisonMode, setComparisonMode] = useState(false);

  // Historical data for comparison view
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [historicalLoading, setHistoricalLoading] = useState(false);
  const [subsequentReturn, setSubsequentReturn] = useState<number | null>(null);

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

  // Fetch historical data for comparison
  const fetchHistoricalData = useCallback(async (pattern: AutoPattern) => {
    setHistoricalLoading(true);
    try {
      const response = await fetch('/api/v1/analysis/historical-kline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: pattern.symbol,
          market: pattern.market,
          start_date: pattern.start_date,
          end_date: pattern.end_date,
          extend_days: 180,
          lookback_days: 180,
        }),
      });
      if (response.ok) {
        const data = await response.json();
        // Convert API format (time) to chart format (date)
        const convertedData = (data.kline_data || []).map((item: any) => ({
          date: item.time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
          volume: item.volume || 0,
        }));
        setHistoricalData(convertedData);
        setSubsequentReturn(data.subsequent_return);
      }
    } catch (err) {
      console.error('Failed to fetch historical data:', err);
    } finally {
      setHistoricalLoading(false);
    }
  }, []);

  // Handlers
  const handleSymbolChange = useCallback((newSymbol: string) => {
    setSymbol(newSymbol);
    setSelection(null);
    clearResults();
    setComparisonMode(false);
  }, [clearResults]);

  const handleMarketChange = useCallback((newMarket: string) => {
    setMarket(newMarket);
    setSelection(null);
    clearResults();
    setComparisonMode(false);
  }, [clearResults]);

  const handleSearch = useCallback(async (request: SearchRequest) => {
    await search(request);
  }, [search]);

  const handlePatternSelect = useCallback((pattern: SimilarPattern) => {
    setSelectedPattern(pattern);
    setAutoSelectedPattern(null);
    setComparisonMode(true);
  }, []);

  const handleAutoPatternSelect = useCallback((pattern: AutoPattern) => {
    setAutoSelectedPattern(pattern);
    setSelectedPattern(null);
    setComparisonMode(true);
    fetchHistoricalData(pattern);
  }, [fetchHistoricalData]);

  const handleCloseComparison = useCallback(() => {
    setComparisonMode(false);
    setSelectedPattern(null);
    setAutoSelectedPattern(null);
    setHistoricalData([]);
  }, []);

  const formatPercent = (value: number | null | undefined) => {
    if (value == null) return '-';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(2)}%`;
  };

  // Comparison View (Side by Side)
  if (comparisonMode && autoSelectedPattern) {
    return (
      <div className="min-h-screen bg-background">
        {/* Comparison Header */}
        <div className="border-b border-border bg-surface">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={handleCloseComparison}
                  className="flex items-center gap-2 px-3 py-2 bg-background border border-border rounded hover:bg-border transition-colors"
                >
                  <ArrowLeft size={16} />
                  返回
                </button>
                <div>
                  <h2 className="text-xl font-bold">形态对比分析</h2>
                  <p className="text-sm text-secondary">
                    相似度: <span className="text-accent font-mono">{(autoSelectedPattern.similarity_score * 100).toFixed(1)}%</span>
                  </p>
                </div>
              </div>
              <button
                onClick={handleCloseComparison}
                className="p-2 hover:bg-border rounded-full transition-colors"
              >
                <X size={20} />
              </button>
            </div>
          </div>
        </div>

        {/* Side by Side Charts */}
        <div className="container mx-auto px-4 py-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Panel - Current Stock */}
            <div className="bg-surface border border-border rounded-lg overflow-hidden">
              <div className="p-4 border-b border-border">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  当前股票
                  <span className="text-xs bg-accent/20 text-accent px-2 py-0.5 rounded">
                    {market.toUpperCase()}
                  </span>
                </h3>
                <p className="text-sm text-secondary mt-1">
                  {symbol} • 最近一年走势
                </p>
              </div>
              <div className="p-4">
                {dataLoading ? (
                  <div className="h-[400px] flex items-center justify-center">
                    <Loader2 className="animate-spin text-accent" size={32} />
                  </div>
                ) : (
                  <KLineChart
                    data={data}
                    selection={null}
                    onSelectionChange={() => { }}
                    height={400}
                    showVolume={false}
                    visibleBarCount={(() => {
                      // Calculate pattern length in days for zoom
                      const startDate = new Date(autoSelectedPattern.start_date);
                      const endDate = new Date(autoSelectedPattern.end_date);
                      const patternDays = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
                      // Add some context (about 30% extra on each side)
                      return Math.max(patternDays + Math.ceil(patternDays * 0.3), 20);
                    })()}
                    highlightDateRange={autoSelectedPattern.current_start_date && autoSelectedPattern.current_end_date ? {
                      startDate: autoSelectedPattern.current_start_date,
                      endDate: autoSelectedPattern.current_end_date
                    } : null}
                  />
                )}
              </div>
            </div>

            {/* Right Panel - Historical Pattern */}
            <div className="bg-surface border border-border rounded-lg overflow-hidden">
              <div className="p-4 border-b border-border">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  历史相似形态
                  <span className="text-xs bg-accent/20 text-accent px-2 py-0.5 rounded">
                    {autoSelectedPattern.market.toUpperCase()}
                  </span>
                </h3>
                <p className="text-sm text-secondary mt-1">
                  {autoSelectedPattern.symbol} • {autoSelectedPattern.start_date} ~ {autoSelectedPattern.end_date}
                </p>
              </div>
              <div className="p-4">
                {historicalLoading ? (
                  <div className="h-[400px] flex items-center justify-center">
                    <Loader2 className="animate-spin text-accent" size={32} />
                    <span className="ml-3 text-secondary">加载历史数据...</span>
                  </div>
                ) : historicalData.length > 0 ? (
                  <KLineChart
                    data={historicalData}
                    selection={null}
                    onSelectionChange={() => { }}
                    height={400}
                    showVolume={false}
                    highlightDateRange={{
                      startDate: autoSelectedPattern.start_date,
                      endDate: autoSelectedPattern.end_date
                    }}
                  />
                ) : (
                  <div className="h-[400px] flex items-center justify-center text-secondary">
                    无法加载历史数据
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Statistics */}
          <div className="mt-6 bg-surface border border-border rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">后续收益统计</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="bg-background p-4 rounded-lg">
                <div className="text-xs text-secondary mb-1">相似度</div>
                <div className="text-xl font-mono text-accent">
                  {(autoSelectedPattern.similarity_score * 100).toFixed(1)}%
                </div>
              </div>
              <div className="bg-background p-4 rounded-lg">
                <div className="text-xs text-secondary mb-1">T+1 收益</div>
                <div className={`text-xl font-mono flex items-center gap-1 ${(autoSelectedPattern.subsequent_returns["t+1"] ?? 0) >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                  {(autoSelectedPattern.subsequent_returns["t+1"] ?? 0) >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                  {formatPercent(autoSelectedPattern.subsequent_returns["t+1"])}
                </div>
              </div>
              <div className="bg-background p-4 rounded-lg">
                <div className="text-xs text-secondary mb-1">T+5 收益</div>
                <div className={`text-xl font-mono flex items-center gap-1 ${(autoSelectedPattern.subsequent_returns["t+5"] ?? 0) >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                  {(autoSelectedPattern.subsequent_returns["t+5"] ?? 0) >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                  {formatPercent(autoSelectedPattern.subsequent_returns["t+5"])}
                </div>
              </div>
              <div className="bg-background p-4 rounded-lg">
                <div className="text-xs text-secondary mb-1">T+10 收益</div>
                <div className={`text-xl font-mono flex items-center gap-1 ${(autoSelectedPattern.subsequent_returns["t+10"] ?? 0) >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                  {(autoSelectedPattern.subsequent_returns["t+10"] ?? 0) >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                  {formatPercent(autoSelectedPattern.subsequent_returns["t+10"])}
                </div>
              </div>
              <div className="bg-background p-4 rounded-lg">
                <div className="text-xs text-secondary mb-1">T+20 收益</div>
                <div className={`text-xl font-mono flex items-center gap-1 ${(autoSelectedPattern.subsequent_returns["t+20"] ?? 0) >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                  {(autoSelectedPattern.subsequent_returns["t+20"] ?? 0) >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                  {formatPercent(autoSelectedPattern.subsequent_returns["t+20"])}
                </div>
              </div>
            </div>
            {subsequentReturn !== null && (
              <div className="mt-4 p-4 bg-background rounded-lg">
                <div className="text-sm text-secondary mb-1">后续30日总收益</div>
                <div className={`text-2xl font-mono flex items-center gap-2 ${subsequentReturn >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                  {subsequentReturn >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
                  {formatPercent(subsequentReturn)}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Normal View
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
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${activeTab === 'auto'
                  ? 'bg-accent text-background'
                  : 'text-secondary hover:text-primary'
                  }`}
              >
                <Zap size={16} />
                自动分析
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${activeTab === 'search'
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
              <AutoAnalysis
                symbol={symbol}
                market={market}
                onPatternSelect={handleAutoPatternSelect}
              />
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
