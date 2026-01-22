import { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  Time,
  MouseEventParams,
  CrosshairMode,
} from 'lightweight-charts';
import type { OHLCVData, ChartSelection } from '../types';

interface KLineChartProps {
  data: OHLCVData[];
  onSelectionChange?: (selection: ChartSelection | null) => void;
  selection?: ChartSelection | null;
  height?: number;
  showVolume?: boolean;
}

export function KLineChart({
  data,
  onSelectionChange,
  selection,
  height = 400,
  showVolume = true,
}: KLineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionStart, setSelectionStart] = useState<number | null>(null);

  // Convert data to chart format
  const candleData: CandlestickData[] = data.map((d) => ({
    time: d.date as Time,
    open: d.open,
    high: d.high,
    low: d.low,
    close: d.close,
  }));

  const volumeData: HistogramData[] = data.map((d) => ({
    time: d.date as Time,
    value: d.volume,
    color: d.close >= d.open ? '#26a69a80' : '#ef535080',
  }));

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: '#0d1117' },
        textColor: '#c9d1d9',
      },
      grid: {
        vertLines: { color: '#30363d' },
        horzLines: { color: '#30363d' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#58a6ff',
          width: 1,
          style: 2,
          labelBackgroundColor: '#58a6ff',
        },
        horzLine: {
          color: '#58a6ff',
          width: 1,
          style: 2,
          labelBackgroundColor: '#58a6ff',
        },
      },
      timeScale: {
        borderColor: '#30363d',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#30363d',
      },
      width: containerRef.current.clientWidth,
      height: height,
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    });

    // Volume series
    let volumeSeries: ISeriesApi<'Histogram'> | null = null;
    if (showVolume) {
      volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
      });

      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      });
    }

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    // Handle resize
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height, showVolume]);

  // Update data
  useEffect(() => {
    if (candleSeriesRef.current && candleData.length > 0) {
      candleSeriesRef.current.setData(candleData);
    }
    if (volumeSeriesRef.current && volumeData.length > 0) {
      volumeSeriesRef.current.setData(volumeData);
    }
    if (chartRef.current && candleData.length > 0) {
      chartRef.current.timeScale().fitContent();
    }
  }, [data]);

  // Handle selection
  const handleMouseDown = useCallback(
    (param: MouseEventParams) => {
      if (!param.time || !onSelectionChange) return;

      setIsSelecting(true);
      const timeStr = param.time.toString();
      const index = data.findIndex((d) => d.date === timeStr);
      setSelectionStart(index >= 0 ? index : null);
    },
    [data, onSelectionChange]
  );

  const handleMouseMove = useCallback(
    (param: MouseEventParams) => {
      if (!isSelecting || selectionStart === null || !param.time) return;

      const timeStr = param.time.toString();
      const currentIndex = data.findIndex((d) => d.date === timeStr);

      if (currentIndex >= 0) {
        const startIdx = Math.min(selectionStart, currentIndex);
        const endIdx = Math.max(selectionStart, currentIndex);

        onSelectionChange?.({
          startIndex: startIdx,
          endIndex: endIdx,
          startDate: data[startIdx].date,
          endDate: data[endIdx].date,
        });
      }
    },
    [isSelecting, selectionStart, data, onSelectionChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsSelecting(false);
  }, []);

  // Subscribe to chart events
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onSelectionChange) return;

    chart.subscribeClick(handleMouseDown);
    chart.subscribeCrosshairMove(handleMouseMove);

    const container = containerRef.current;
    if (container) {
      container.addEventListener('mouseup', handleMouseUp);
      container.addEventListener('mouseleave', handleMouseUp);
    }

    return () => {
      chart.unsubscribeClick(handleMouseDown);
      chart.unsubscribeCrosshairMove(handleMouseMove);
      if (container) {
        container.removeEventListener('mouseup', handleMouseUp);
        container.removeEventListener('mouseleave', handleMouseUp);
      }
    };
  }, [handleMouseDown, handleMouseMove, handleMouseUp, onSelectionChange]);

  // Render selection overlay
  const renderSelectionOverlay = () => {
    if (!selection || !chartRef.current || !containerRef.current) return null;

    const timeScale = chartRef.current.timeScale();
    const startX = timeScale.timeToCoordinate(data[selection.startIndex].date as Time);
    const endX = timeScale.timeToCoordinate(data[selection.endIndex].date as Time);

    if (startX === null || endX === null) return null;

    const left = Math.min(startX, endX);
    const width = Math.abs(endX - startX);

    return (
      <div
        className="selection-overlay"
        style={{
          left: `${left}px`,
          width: `${width}px`,
        }}
      />
    );
  };

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="chart-container"
        style={{ height: `${height}px` }}
      />
      {renderSelectionOverlay()}
      {selection && (
        <div className="absolute top-2 right-2 bg-surface border border-border rounded px-3 py-1 text-sm">
          <span className="text-secondary">Selected: </span>
          <span className="text-accent">
            {selection.endIndex - selection.startIndex + 1} days
          </span>
          <span className="text-muted ml-2">
            ({selection.startDate} - {selection.endDate})
          </span>
        </div>
      )}
    </div>
  );
}

export default KLineChart;
