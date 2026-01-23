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
  visibleBarCount?: number;
  highlightDateRange?: { startDate: string; endDate: string } | null;
}

export function KLineChart({
  data,
  onSelectionChange,
  selection,
  height = 400,
  showVolume = true,
  visibleBarCount,
  highlightDateRange,
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

  // Update data and markers
  useEffect(() => {
    if (candleSeriesRef.current && candleData.length > 0) {
      candleSeriesRef.current.setData(candleData);

      // Variables to store pattern range indices for zoom
      let patternStartIdx = -1;
      let patternEndIdx = -1;

      // Add markers for highlighted range (start/end arrows)
      if (highlightDateRange) {
        // Find indices using >= and <= matching for robustness
        let startIdx = data.findIndex(d => d.date >= highlightDateRange.startDate);
        let endIdx = -1;
        for (let i = data.length - 1; i >= 0; i--) {
          if (data[i].date <= highlightDateRange.endDate) {
            endIdx = i;
            break;
          }
        }

        // Store for zoom logic
        patternStartIdx = startIdx;
        patternEndIdx = endIdx;

        const markers: any[] = [];

        if (startIdx >= 0 && startIdx < data.length) {
          markers.push({
            time: data[startIdx].date as Time,
            position: 'belowBar',
            color: '#f59e0b',
            shape: 'arrowUp',
            text: '起点',
          });
        }

        if (endIdx >= 0) {
          markers.push({
            time: data[endIdx].date as Time,
            position: 'aboveBar',
            color: '#ef4444',
            shape: 'arrowDown',
            text: '终点(现在)',
          });
        }

        candleSeriesRef.current.setMarkers(markers);
      } else {
        candleSeriesRef.current.setMarkers([]);
      }

      if (volumeSeriesRef.current && volumeData.length > 0) {
        volumeSeriesRef.current.setData(volumeData);
      }

      if (chartRef.current && candleData.length > 0) {
        const timeScale = chartRef.current.timeScale();

        // If we have a highlight range, set view based on pattern
        if (highlightDateRange && patternStartIdx >= 0 && patternEndIdx >= 0) {
          const patternLength = patternEndIdx - patternStartIdx;
          // Add padding (approx 15% on each side to match left chart's ~30% total context)
          const padding = Math.max(5, Math.ceil(patternLength * 0.15));

          timeScale.setVisibleLogicalRange({
            from: patternStartIdx - padding,
            to: patternEndIdx + padding,
          });
        } else if (visibleBarCount && visibleBarCount > 0) {
          timeScale.setVisibleLogicalRange({
            from: candleData.length - visibleBarCount,
            to: candleData.length,
          });
        } else {
          timeScale.fitContent();
        }
      }
    }
  }, [data, visibleBarCount, highlightDateRange]);

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

  const renderHighlightOverlay = () => {
    if (!highlightDateRange || !chartRef.current || !containerRef.current || data.length === 0) return null;

    const timeScale = chartRef.current.timeScale();
    const startX = timeScale.timeToCoordinate(highlightDateRange.startDate as Time);
    const endX = timeScale.timeToCoordinate(highlightDateRange.endDate as Time);

    // If coordinates are null, it might be off-screen or not loaded, handle gracefully?
    // We try to find closest indices if exact dates not found
    // But chart format conversion might have handled exact dates.

    if (startX === null && endX === null) return null;

    // Simple fallback if one bound is visible
    // Wait, timeToCoordinate returns null if the point is arguably valid? No, it returns coordinate.
    // If null, effectively don't draw.
    // Actually we need indices to be robust.

    const startIndex = data.findIndex(d => d.date >= highlightDateRange.startDate);
    const endIndex = data.findIndex(d => d.date > highlightDateRange.endDate) - 1;

    // Re-check coordinates based on robust indices
    const robustStartIndex = startIndex >= 0 ? startIndex : 0;
    const robustEndIndex = endIndex >= 0 ? endIndex : data.length - 1;

    // Actually timeToCoordinate is best if we trust the dates exist.
    // Let's stick to timeToCoordinate but handle nulls by clamping to chart edges if needed?
    // For now simple optional logic.

    const x1 = startX ?? timeScale.timeToCoordinate(data[robustStartIndex].date as Time);
    const x2 = endX ?? timeScale.timeToCoordinate(data[robustEndIndex].date as Time);

    if (x1 === null || x2 === null) return null;

    const left = Math.min(x1, x2);
    const width = Math.abs(x2 - x1);

    return (
      <div
        className="highlight-overlay"
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          left: `${left}px`,
          width: `${width}px`,
          backgroundColor: 'rgba(64, 158, 255, 0.1)', // Blue-ish tint
          borderLeft: '1px dashed rgba(64, 158, 255, 0.5)',
          borderRight: '1px dashed rgba(64, 158, 255, 0.5)',
          pointerEvents: 'none',
          zIndex: 5,
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
      {renderHighlightOverlay()}
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
