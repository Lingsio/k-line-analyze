import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';
import type { AnalysisSummary, SearchResponse } from '../types';

interface AnalysisPanelProps {
  results: SearchResponse | null;
}

export function AnalysisPanel({ results }: AnalysisPanelProps) {
  if (!results) {
    return null;
  }

  const { analysis } = results;

  // Prepare data for win rate chart
  const winRateData = [
    { period: 'T+1', rate: analysis.win_rate_1d * 100 },
    { period: 'T+5', rate: analysis.win_rate_5d * 100 },
    { period: 'T+20', rate: analysis.win_rate_20d * 100 },
  ];

  // Prepare data for return chart
  const returnData = [
    { period: 'T+1', return: analysis.avg_return_1d * 100 },
    { period: 'T+5', return: analysis.avg_return_5d * 100 },
    { period: 'T+20', return: analysis.avg_return_20d * 100 },
  ];

  const getSignalStrength = () => {
    const winRate = analysis.win_rate_5d;
    if (winRate > 0.65) return { text: 'Strong Bullish', color: 'text-bullish', icon: TrendingUp };
    if (winRate > 0.55) return { text: 'Moderately Bullish', color: 'text-bullish', icon: TrendingUp };
    if (winRate > 0.45) return { text: 'Neutral', color: 'text-secondary', icon: Target };
    if (winRate > 0.35) return { text: 'Moderately Bearish', color: 'text-bearish', icon: TrendingDown };
    return { text: 'Strong Bearish', color: 'text-bearish', icon: TrendingDown };
  };

  const signal = getSignalStrength();
  const SignalIcon = signal.icon;

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-4">Statistical Analysis</h3>

      {/* Signal Summary */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-background rounded-lg p-4">
          <div className="text-sm text-secondary mb-2">Signal</div>
          <div className={`flex items-center gap-2 text-xl font-semibold ${signal.color}`}>
            <SignalIcon size={24} />
            {signal.text}
          </div>
        </div>
        <div className="bg-background rounded-lg p-4">
          <div className="text-sm text-secondary mb-2">Confidence</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-border rounded-full overflow-hidden">
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${analysis.confidence * 100}%` }}
              />
            </div>
            <span className="font-mono text-accent">
              {(analysis.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-background rounded p-3 text-center">
          <div className="text-2xl font-bold text-accent">
            {analysis.total_matches}
          </div>
          <div className="text-xs text-secondary">Matches</div>
        </div>
        <div className="bg-background rounded p-3 text-center">
          <div className={`text-2xl font-bold ${analysis.win_rate_5d > 0.5 ? 'text-bullish' : 'text-bearish'}`}>
            {(analysis.win_rate_5d * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-secondary">Win Rate (T+5)</div>
        </div>
        <div className="bg-background rounded p-3 text-center">
          <div className={`text-2xl font-bold ${analysis.avg_return_5d >= 0 ? 'text-bullish' : 'text-bearish'}`}>
            {formatPercent(analysis.avg_return_5d * 100)}
          </div>
          <div className="text-xs text-secondary">Avg Return (T+5)</div>
        </div>
      </div>

      {/* Win Rate Chart */}
      <div className="mb-6">
        <div className="text-sm text-secondary mb-2">Win Rate by Period</div>
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={winRateData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                stroke="#8b949e"
                fontSize={12}
              />
              <YAxis
                type="category"
                dataKey="period"
                stroke="#8b949e"
                fontSize={12}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '4px',
                }}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Win Rate']}
              />
              <Bar dataKey="rate" radius={[0, 4, 4, 0]}>
                {winRateData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.rate >= 50 ? '#26a69a' : '#ef5350'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Average Returns Chart */}
      <div className="mb-6">
        <div className="text-sm text-secondary mb-2">Average Returns by Period</div>
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={returnData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                type="number"
                tickFormatter={(v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`}
                stroke="#8b949e"
                fontSize={12}
              />
              <YAxis
                type="category"
                dataKey="period"
                stroke="#8b949e"
                fontSize={12}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#161b22',
                  border: '1px solid #30363d',
                  borderRadius: '4px',
                }}
                formatter={(value: number) => [
                  `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`,
                  'Avg Return',
                ]}
              />
              <Bar dataKey="return" radius={[0, 4, 4, 0]}>
                {returnData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.return >= 0 ? '#26a69a' : '#ef5350'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 p-3 bg-background rounded border border-border text-xs text-secondary">
        <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
        <div>
          Past performance does not guarantee future results. This analysis is for
          informational purposes only and should not be considered financial advice.
        </div>
      </div>
    </div>
  );
}

export default AnalysisPanel;
