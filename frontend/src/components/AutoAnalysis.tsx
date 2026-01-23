import { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart2,
  Zap,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Sparkles,
  Loader2,
} from 'lucide-react';

interface SimilarPatternDetail {
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

interface TimeframeAnalysis {
  timeframe: string;
  window_days: number;
  start_date: string;
  end_date: string;
  period: string;
  signal: string;
  win_rate_5d: number;
  win_rate_20d: number;
  avg_return_5d: number;
  avg_return_20d: number;
  confidence: number;
  similar_count: number;
  similar_patterns: SimilarPatternDetail[];
}

interface AutoAnalysisData {
  symbol: string;
  market: string;
  analysis_date: string;
  overall_signal: string;
  overall_confidence: number;
  recommendation: string;
  risk_level: string;
  key_insights: string[];
  llm_analysis?: string | null;
  timeframe_analyses: TimeframeAnalysis[];
}

interface AutoAnalysisProps {
  symbol: string;
  market: string;
  onPatternSelect?: (pattern: SimilarPatternDetail) => void;
}

const SIGNAL_CONFIG = {
  strong_bullish: {
    icon: TrendingUp,
    color: 'text-bullish',
    bg: 'bg-bullish/10',
    border: 'border-bullish/30',
    label: '强势看涨',
  },
  bullish: {
    icon: TrendingUp,
    color: 'text-bullish',
    bg: 'bg-bullish/10',
    border: 'border-bullish/20',
    label: '偏多',
  },
  neutral: {
    icon: Minus,
    color: 'text-secondary',
    bg: 'bg-secondary/10',
    border: 'border-secondary/20',
    label: '中性',
  },
  bearish: {
    icon: TrendingDown,
    color: 'text-bearish',
    bg: 'bg-bearish/10',
    border: 'border-bearish/20',
    label: '偏空',
  },
  strong_bearish: {
    icon: TrendingDown,
    color: 'text-bearish',
    bg: 'bg-bearish/10',
    border: 'border-bearish/30',
    label: '强势看跌',
  },
};

const RISK_CONFIG = {
  low: { color: 'text-bullish', label: '低', icon: CheckCircle },
  medium: { color: 'text-yellow-500', label: '中', icon: AlertTriangle },
  high: { color: 'text-bearish', label: '高', icon: AlertTriangle },
};

const TIMEFRAME_LABELS: Record<string, string> = {
  daily_20: '日K (20天)',
  daily_60: '日K (60天)',
  weekly_12: '周K (12周)',
  weekly_26: '周K (26周)',
  monthly_6: '月K (6月)',
  monthly_12: '月K (12月)',
};

export function AutoAnalysis({ symbol, market, onPatternSelect }: AutoAnalysisProps) {
  const [data, setData] = useState<AutoAnalysisData | null>(null);
  const [loading, setLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedTimeframe, setExpandedTimeframe] = useState<string | null>(null);

  const fetchAnalysis = async (useLlm: boolean = false) => {
    if (!symbol) return;

    if (useLlm) {
      setAiLoading(true);
    } else {
      setLoading(true);
    }
    setError(null);

    try {
      const params = new URLSearchParams({
        market,
        self_only: 'true',
        use_llm: useLlm.toString(),
      });

      const response = await fetch(
        `/api/v1/analysis/auto/${symbol}?${params}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch analysis');
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
      setAiLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysis(false);
  }, [symbol, market]);

  const toggleExpand = (timeframe: string) => {
    if (expandedTimeframe === timeframe) {
      setExpandedTimeframe(null);
    } else {
      setExpandedTimeframe(timeframe);
    }
  };

  const getSignalConfig = (signal: string) => {
    return SIGNAL_CONFIG[signal as keyof typeof SIGNAL_CONFIG] || SIGNAL_CONFIG.neutral;
  };

  const getRiskConfig = (risk: string) => {
    return RISK_CONFIG[risk as keyof typeof RISK_CONFIG] || RISK_CONFIG.medium;
  };

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${(value * 100).toFixed(2)}%`;
  };

  const handlePatternClick = (pattern: SimilarPatternDetail, tf: TimeframeAnalysis) => {
    if (onPatternSelect) {
      // Merge current stock's pattern interval with historical pattern
      const enrichedPattern = {
        ...pattern,
        current_start_date: tf.start_date,
        current_end_date: tf.end_date,
      };
      onPatternSelect(enrichedPattern);
    }
  };

  if (loading) {
    return (
      <div className="bg-surface border border-border rounded-lg p-6">
        <div className="flex items-center justify-center gap-3">
          <RefreshCw className="animate-spin text-accent" size={24} />
          <span className="text-secondary">正在分析 {symbol} 的走势...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-surface border border-border rounded-lg p-6">
        <div className="text-center text-bearish">
          <AlertTriangle size={24} className="mx-auto mb-2" />
          <p>{error}</p>
          <button
            onClick={() => fetchAnalysis()}
            className="mt-3 px-4 py-2 bg-accent text-background rounded hover:bg-accent/90"
          >
            重试
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-surface border border-border rounded-lg p-6 text-center text-secondary">
        <BarChart2 size={32} className="mx-auto mb-2 opacity-50" />
        <p>输入股票代码后自动分析近期走势</p>
      </div>
    );
  }

  const overallConfig = getSignalConfig(data.overall_signal);
  const riskConfig = getRiskConfig(data.risk_level);
  const OverallIcon = overallConfig.icon;
  const RiskIcon = riskConfig.icon;

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap size={20} className="text-accent" />
          <h3 className="font-semibold">自动分析</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchAnalysis(true)}
            disabled={aiLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 rounded transition-colors disabled:opacity-50"
            title="AI智能分析当前K线形态"
          >
            {aiLoading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Sparkles size={14} />
            )}
            <span className="text-sm">{aiLoading ? 'AI分析中...' : 'AI分析'}</span>
          </button>
          <button
            onClick={() => fetchAnalysis(false)}
            className="p-2 hover:bg-border rounded transition-colors"
            title="刷新分析"
          >
            <RefreshCw size={16} className="text-secondary" />
          </button>
        </div>
      </div>

      {/* Overall Signal Card */}
      <div className={`m-4 p-4 rounded-lg border ${overallConfig.bg} ${overallConfig.border}`}>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-secondary mb-1">综合信号</div>
            <div className={`text-2xl font-bold flex items-center gap-2 ${overallConfig.color}`}>
              <OverallIcon size={28} />
              {overallConfig.label}
            </div>
          </div>
        </div>

        {/* Risk Level */}
        <div className="mt-3 pt-3 border-t border-border/50 flex items-center gap-2">
          <RiskIcon size={16} className={riskConfig.color} />
          <span className="text-sm text-secondary">风险等级：</span>
          <span className={`font-medium ${riskConfig.color}`}>{riskConfig.label}</span>
        </div>
      </div>

      {/* Recommendation */}
      <div className="mx-4 mb-4 p-3 bg-background rounded-lg">
        <div className="flex items-start gap-2">
          <Target size={18} className="text-accent flex-shrink-0 mt-0.5" />
          <p className="text-sm">{data.recommendation}</p>
        </div>
      </div>

      {/* LLM Analysis (if available) */}
      {data.llm_analysis && (
        <div className="mx-4 mb-4 p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={16} className="text-purple-400" />
            <span className="text-sm font-medium text-purple-400">AI 智能分析</span>
          </div>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{data.llm_analysis}</p>
        </div>
      )}

      {/* Key Insights */}
      <div className="mx-4 mb-4">
        <div className="text-sm text-secondary mb-2 flex items-center gap-1">
          <CheckCircle size={14} />
          关键洞察
        </div>
        <ul className="space-y-1">
          {data.key_insights.map((insight, idx) => (
            <li key={idx} className="text-sm flex items-start gap-2">
              <span className="text-accent">•</span>
              <span>{insight}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Timeframe Analysis Grid */}
      <div className="border-t border-border">
        <div className="p-4">
          <div className="text-sm text-secondary mb-3 flex items-center gap-1">
            <Clock size={14} />
            多周期分析
          </div>

          <div className="space-y-2">
            {data.timeframe_analyses.map((tf) => {
              const tfConfig = getSignalConfig(tf.signal);
              const TfIcon = tfConfig.icon;
              const isExpanded = expandedTimeframe === tf.timeframe;

              return (
                <div key={tf.timeframe} className="bg-background rounded-lg overflow-hidden border border-transparent hover:border-border transition-colors">
                  <div
                    className="flex items-center justify-between p-3 cursor-pointer"
                    onClick={() => toggleExpand(tf.timeframe)}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-1.5 rounded ${tfConfig.bg}`}>
                        <TfIcon size={16} className={tfConfig.color} />
                      </div>
                      <div>
                        <div className="font-medium text-sm flex items-center gap-2">
                          {TIMEFRAME_LABELS[tf.timeframe] || tf.timeframe}
                          {isExpanded ? <ChevronUp size={14} className="text-secondary" /> : <ChevronDown size={14} className="text-secondary" />}
                        </div>
                        <div className="text-xs text-secondary">
                          {tf.similar_count} 个相似形态
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-right">
                      <div>
                        <div className="text-xs text-secondary">5日胜率</div>
                        <div className={`font-mono text-sm ${tf.win_rate_5d > 0.5 ? 'text-bullish' : 'text-bearish'}`}>
                          {(tf.win_rate_5d * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-secondary">5日收益</div>
                        <div className={`font-mono text-sm ${tf.avg_return_5d >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                          {formatPercent(tf.avg_return_5d)}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Expanded Details: Similar Patterns List */}
                  {isExpanded && (
                    <div className="border-t border-border bg-surface/50 p-3">
                      <div className="text-xs font-semibold text-secondary mb-2 uppercase tracking-wider">
                        相似历史形态 <span className="text-accent">(点击查看对比图表)</span>
                      </div>
                      <div className="space-y-2">
                        {tf.similar_patterns && tf.similar_patterns.map((pattern, idx) => (
                          <div
                            key={idx}
                            className="grid grid-cols-4 gap-2 text-xs p-2 bg-background rounded border border-border/50 hover:bg-accent/10 hover:border-accent/30 transition-colors items-center cursor-pointer"
                            onClick={() => handlePatternClick(pattern, tf)}
                          >
                            <div className="col-span-1 font-medium flex items-center gap-1">
                              <ExternalLink size={10} className="text-accent" />
                              {pattern.symbol} ({pattern.market})
                            </div>
                            <div className="col-span-1 text-secondary text-right">
                              {pattern.end_date}
                            </div>
                            <div className="col-span-1 text-right">
                              <span className="text-secondary mr-1">T+5:</span>
                              <span className={pattern.subsequent_returns["t+5"] >= 0 ? "text-bullish" : "text-bearish"}>
                                {pattern.subsequent_returns["t+5"] != null ? formatPercent(pattern.subsequent_returns["t+5"]) : '-'}
                              </span>
                            </div>
                            <div className="col-span-1 text-right">
                              <span className="text-secondary mr-1">T+20:</span>
                              <span className={pattern.subsequent_returns["t+20"] >= 0 ? "text-bullish" : "text-bearish"}>
                                {pattern.subsequent_returns["t+20"] != null ? formatPercent(pattern.subsequent_returns["t+20"]) : '-'}
                              </span>
                            </div>
                          </div>
                        ))}
                        {(!tf.similar_patterns || tf.similar_patterns.length === 0) && (
                          <div className="text-center text-xs text-secondary py-2">暂无详细数据</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 bg-background/50 text-xs text-secondary border-t border-border">
        分析时间：{data.analysis_date} | 点击相似形态可查看对比图表
      </div>
    </div>
  );
}

export default AutoAnalysis;
