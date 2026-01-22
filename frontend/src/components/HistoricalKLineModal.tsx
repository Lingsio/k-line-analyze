import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { X, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';

interface KLineData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface PatternInfo {
  symbol: string;
  market: string;
  start_date: string;
  end_date: string;
  similarity_score: number;
  subsequent_returns: {
    "t+1": number;
    "t+5": number;
    "t+10": number;
    "t+20": number;
  };
}

interface HistoricalKLineModalProps {
  pattern: PatternInfo;
  onClose: () => void;
}

export function HistoricalKLineModal({ pattern, onClose }: HistoricalKLineModalProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [klineData, setKlineData] = useState<KLineData[]>([]);
  const [subsequentReturn, setSubsequentReturn] = useState<number | null>(null);

  useEffect(() => {
    fetchHistoricalData();
    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [pattern]);

  const fetchHistoricalData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/analysis/historical-kline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: pattern.symbol,
          market: pattern.market,
          start_date: pattern.start_date,
          end_date: pattern.end_date,
          extend_days: 30,
        }),
      });

      if (!response.ok) {
        throw new Error('获取历史数据失败');
      }

      const data = await response.json();
      setKlineData(data.kline_data);
      setSubsequentReturn(data.subsequent_return);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!chartContainerRef.current || loading || klineData.length === 0) return;

    // 创建图表
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#1a1a2e' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        borderColor: '#2B2B43',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#2B2B43',
      },
      crosshair: {
        mode: 1,
      },
    });

    chartRef.current = chart;

    // 添加K线系列
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    candleSeriesRef.current = candleSeries;

    // 转换并设置数据
    const chartData: CandlestickData<Time>[] = klineData.map((d) => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    candleSeries.setData(chartData);

    // 高亮显示形态期间
    // 使用标记来表示形态范围
    const markers: any[] = [];
    const startIndex = klineData.findIndex(d => d.time === pattern.start_date);
    const endIndex = klineData.findIndex(d => d.time === pattern.end_date);

    if (startIndex >= 0) {
      markers.push({
        time: klineData[startIndex].time as Time,
        position: 'belowBar',
        color: '#00bcd4',
        shape: 'arrowUp',
        text: '形态开始',
      });
    }

    if (endIndex >= 0) {
      markers.push({
        time: klineData[endIndex].time as Time,
        position: 'aboveBar',
        color: '#ff9800',
        shape: 'arrowDown',
        text: '形态结束',
      });
    }

    candleSeries.setMarkers(markers);

    // 自适应
    chart.timeScale().fitContent();

    // 窗口大小变化时调整
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [klineData, loading]);

  const formatPercent = (value: number | null | undefined) => {
    if (value == null) return '-';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <div className="bg-surface border border-border rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h3 className="text-lg font-semibold flex items-center gap-2">
              {pattern.symbol}
              <span className="text-xs bg-accent/20 text-accent px-2 py-0.5 rounded">
                {pattern.market.toUpperCase()}
              </span>
            </h3>
            <p className="text-sm text-secondary">
              历史形态：{pattern.start_date} ~ {pattern.end_date}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-border rounded-full transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center h-[400px]">
              <Loader2 className="animate-spin text-accent" size={32} />
              <span className="ml-3 text-secondary">加载历史数据...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-[400px] text-bearish">
              <p>{error}</p>
            </div>
          ) : (
            <>
              {/* Chart */}
              <div ref={chartContainerRef} className="w-full rounded-lg overflow-hidden" />

              {/* Stats */}
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-background p-3 rounded-lg">
                  <div className="text-xs text-secondary mb-1">相似度</div>
                  <div className="text-lg font-mono text-accent">
                    {(pattern.similarity_score * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="bg-background p-3 rounded-lg">
                  <div className="text-xs text-secondary mb-1">T+5 收益</div>
                  <div className={`text-lg font-mono flex items-center gap-1 ${
                    pattern.subsequent_returns?.["t+5"] >= 0 ? 'text-bullish' : 'text-bearish'
                  }`}>
                    {pattern.subsequent_returns?.["t+5"] >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                    {formatPercent(pattern.subsequent_returns?.["t+5"])}
                  </div>
                </div>

                <div className="bg-background p-3 rounded-lg">
                  <div className="text-xs text-secondary mb-1">T+20 收益</div>
                  <div className={`text-lg font-mono flex items-center gap-1 ${
                    pattern.subsequent_returns?.["t+20"] >= 0 ? 'text-bullish' : 'text-bearish'
                  }`}>
                    {pattern.subsequent_returns?.["t+20"] >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                    {formatPercent(pattern.subsequent_returns?.["t+20"])}
                  </div>
                </div>

                <div className="bg-background p-3 rounded-lg">
                  <div className="text-xs text-secondary mb-1">后续30日</div>
                  <div className={`text-lg font-mono flex items-center gap-1 ${
                    (subsequentReturn ?? 0) >= 0 ? 'text-bullish' : 'text-bearish'
                  }`}>
                    {(subsequentReturn ?? 0) >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                    {formatPercent(subsequentReturn)}
                  </div>
                </div>
              </div>

              {/* Legend */}
              <div className="mt-4 flex items-center gap-4 text-xs text-secondary">
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-cyan-500 rounded" />
                  <span>形态开始</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-orange-500 rounded" />
                  <span>形态结束</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-green-500 rounded" />
                  <span>阳线</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-3 h-3 bg-red-500 rounded" />
                  <span>阴线</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-border text-xs text-secondary text-center">
          图表展示了该历史形态期间及后续30天的K线走势
        </div>
      </div>
    </div>
  );
}

export default HistoricalKLineModal;
