from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel, Field

router = APIRouter()


class ReturnDistribution(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    percentage: float


class DetailedAnalysis(BaseModel):
    period: str  # "T+1", "T+5", "T+20"
    sample_size: int
    win_rate: float
    avg_return: float
    median_return: float
    std_deviation: float
    max_return: float
    min_return: float
    percentile_25: float
    percentile_75: float
    distribution: List[ReturnDistribution]


class PatternAnalysisResponse(BaseModel):
    pattern_id: Optional[str] = None
    query_symbol: str
    total_similar_patterns: int
    analyses: List[DetailedAnalysis]
    recommendation: str
    confidence_score: float


@router.get("/analysis/pattern/{pattern_id}", response_model=PatternAnalysisResponse)
async def get_pattern_analysis(
    pattern_id: str,
    lookahead_days: List[int] = Query([1, 5, 10, 20], description="Days to analyze"),
):
    """
    Get detailed analysis for a saved pattern search.

    This provides more detailed statistics than the search endpoint.
    """
    # For now, return mock data - in production, this would fetch from a cache/db
    import random
    import numpy as np

    analyses = []
    for days in lookahead_days:
        # Generate mock return distribution
        returns = [random.gauss(0.02, 0.05) for _ in range(50)]
        bins = np.histogram(returns, bins=10)

        distribution = []
        for i in range(len(bins[0])):
            distribution.append(
                ReturnDistribution(
                    bin_start=round(float(bins[1][i]), 4),
                    bin_end=round(float(bins[1][i + 1]), 4),
                    count=int(bins[0][i]),
                    percentage=round(float(bins[0][i]) / len(returns), 3),
                )
            )

        analyses.append(
            DetailedAnalysis(
                period=f"T+{days}",
                sample_size=len(returns),
                win_rate=round(sum(1 for r in returns if r > 0) / len(returns), 3),
                avg_return=round(float(np.mean(returns)), 4),
                median_return=round(float(np.median(returns)), 4),
                std_deviation=round(float(np.std(returns)), 4),
                max_return=round(max(returns), 4),
                min_return=round(min(returns), 4),
                percentile_25=round(float(np.percentile(returns, 25)), 4),
                percentile_75=round(float(np.percentile(returns, 75)), 4),
                distribution=distribution,
            )
        )

    # Generate recommendation based on win rate
    avg_win_rate = sum(a.win_rate for a in analyses) / len(analyses)
    if avg_win_rate > 0.65:
        recommendation = "Strong bullish signal based on historical patterns"
    elif avg_win_rate > 0.55:
        recommendation = "Moderately bullish signal, consider with other indicators"
    elif avg_win_rate > 0.45:
        recommendation = "Neutral signal, no clear direction from historical patterns"
    else:
        recommendation = "Bearish signal based on historical patterns"

    return PatternAnalysisResponse(
        pattern_id=pattern_id,
        query_symbol="AAPL",
        total_similar_patterns=50,
        analyses=analyses,
        recommendation=recommendation,
        confidence_score=round(0.6 + avg_win_rate * 0.3, 3),
    )


class CompareRequest(BaseModel):
    pattern_ids: List[str] = Field(..., min_length=2, max_length=5)


@router.post("/analysis/compare")
async def compare_patterns(request: CompareRequest):
    """Compare multiple pattern searches."""
    # Placeholder for pattern comparison logic
    return {
        "patterns": request.pattern_ids,
        "comparison": "Feature under development",
    }


@router.get("/analysis/statistics")
async def get_global_statistics():
    """Get global statistics about indexed patterns."""
    return {
        "total_patterns_indexed": 85_000_000,
        "markets": {
            "us": {"stocks": 8000, "patterns": 38_000_000},
            "tw": {"stocks": 2000, "patterns": 9_000_000},
            "cn": {"stocks": 5000, "patterns": 23_000_000},
            "hk": {"stocks": 2500, "patterns": 12_000_000},
            "crypto": {"tokens": 500, "patterns": 3_000_000},
        },
        "time_range": {"start": "2005-01-01", "end": "2025-01-22"},
        "window_sizes_available": [20, 60, 120],
        "model_version": "v1.0.0",
        "last_updated": "2025-01-22",
    }


# ============ 自动分析 API ============

class SimilarPatternDetail(BaseModel):
    """相似形态详细信息"""
    symbol: str
    market: str
    start_date: str
    end_date: str
    similarity_score: float
    subsequent_returns: dict


class TimeframeAnalysisResult(BaseModel):
    """单一时间周期分析结果"""
    timeframe: str
    window_days: int
    period: str
    start_date: str
    end_date: str
    signal: str
    win_rate_5d: float
    win_rate_20d: float
    avg_return_5d: float
    avg_return_20d: float
    confidence: float
    similar_count: int
    similar_patterns: List[SimilarPatternDetail] = []
    confidence: float
    similar_count: int


class AutoAnalysisResponse(BaseModel):
    """自动分析响应"""
    symbol: str
    market: str
    analysis_date: str
    overall_signal: str
    overall_confidence: float
    recommendation: str
    risk_level: str
    key_insights: List[str]
    llm_analysis: Optional[str] = None
    timeframe_analyses: List[TimeframeAnalysisResult]


@router.get("/analysis/auto/{symbol}", response_model=AutoAnalysisResponse)
async def auto_analyze_stock(
    symbol: str,
    market: str = Query("us", description="Market: us, tw, cn, hk, crypto"),
    timeframes: List[str] = Query(
        ["daily", "weekly", "monthly"],
        description="Timeframes to analyze"
    ),
    self_only: bool = Query(
        False,
        description="Only search similar patterns from this stock's own history"
    ),
    use_llm: bool = Query(
        False,
        description="Use LLM for enhanced analysis"
    ),
    force_refresh: bool = Query(
        False,
        description="Force refresh LLM analysis (ignore cache)"
    ),
):
    """
    自动分析股票近期走势

    综合分析该股票的日K、周K、月K走势，通过与历史相似形态对比，
    给出多维度的趋势分析和投资建议。

    **分析内容：**
    - 日K分析：最近20天、60天走势
    - 周K分析：最近12周、26周走势
    - 月K分析：最近6个月、12个月走势

    **返回信息：**
    - 各时间维度的信号方向和胜率
    - 综合信号和置信度
    - 投资建议和风险等级
    - 关键洞察要点
    - LLM智能分析（如启用）
    """
    from app.services.auto_analyzer import quick_analyze, TimeFrame

    try:
        # 转换时间周期参数
        tf_map = {
            "daily": TimeFrame.DAILY,
            "weekly": TimeFrame.WEEKLY,
            "monthly": TimeFrame.MONTHLY,
        }

        result = await quick_analyze(
            symbol=symbol, 
            market=market,
            self_only=self_only,
            use_llm=use_llm,
            force_refresh=force_refresh,
        )

        return AutoAnalysisResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class HistoricalKLineRequest(BaseModel):
    """历史K线请求"""
    symbol: str
    market: str
    start_date: str
    end_date: str
    extend_days: int = 30  # 向后延伸的天数，用于显示后续走势
    lookback_days: int = 30  # 向前延伸的天数，用于显示历史背景


class KLineData(BaseModel):
    """K线数据"""
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class HistoricalKLineResponse(BaseModel):
    """历史K线响应"""
    symbol: str
    market: str
    pattern_period: dict  # {"start": str, "end": str}
    kline_data: List[KLineData]
    subsequent_return: Optional[float] = None


@router.post("/analysis/historical-kline", response_model=HistoricalKLineResponse)
async def get_historical_kline(request: HistoricalKLineRequest):
    """
    获取历史K线数据
    
    用于显示相似形态的历史K线图，包括形态期间和后续走势。
    """
    from app.services.data_fetcher import DataFetcher
    from datetime import datetime, timedelta
    
    try:
        fetcher = DataFetcher()
        
        # 解析日期
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        # 向前多取一些数据作为背景
        fetch_start = start - timedelta(days=request.lookback_days)
        # 向后延伸以显示后续走势
        fetch_end = end + timedelta(days=request.extend_days)
        
        # 获取K线数据
        df = await fetcher.fetch_ohlcv(
            symbol=request.symbol,
            market=request.market,
            start_date=fetch_start.strftime("%Y-%m-%d"),
            end_date=fetch_end.strftime("%Y-%m-%d"),
        )
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found for this symbol and date range")
        
        # 转换为K线数据列表
        kline_data = []
        for idx, row in df.iterrows():
            kline_data.append(KLineData(
                time=idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10],
                open=round(float(row.get('Open', row.get('open', 0))), 4),
                high=round(float(row.get('High', row.get('high', 0))), 4),
                low=round(float(row.get('Low', row.get('low', 0))), 4),
                close=round(float(row.get('Close', row.get('close', 0))), 4),
                volume=float(row.get('Volume', row.get('volume', 0))) if 'Volume' in row or 'volume' in row else None,
            ))
        
        # 计算后续收益
        subsequent_return = None
        try:
            # 找到形态结束日期之后的数据
            end_idx = None
            for i, k in enumerate(kline_data):
                if k.time == request.end_date:
                    end_idx = i
                    break
            
            if end_idx is not None and end_idx < len(kline_data) - 1:
                end_close = kline_data[end_idx].close
                # 取形态结束后20天或最后一天的收盘价
                future_idx = min(end_idx + 20, len(kline_data) - 1)
                future_close = kline_data[future_idx].close
                subsequent_return = (future_close - end_close) / end_close
        except:
            pass
        
        return HistoricalKLineResponse(
            symbol=request.symbol,
            market=request.market,
            pattern_period={"start": request.start_date, "end": request.end_date},
            kline_data=kline_data,
            subsequent_return=subsequent_return,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QuickInsightResponse(BaseModel):
    """快速洞察响应"""
    symbol: str
    market: str
    short_term_signal: str  # 短期信号
    mid_term_signal: str    # 中期信号
    long_term_signal: str   # 长期信号
    overall_signal: str     # 综合信号
    confidence: float
    one_line_summary: str   # 一句话总结


@router.get("/analysis/quick/{symbol}", response_model=QuickInsightResponse)
async def quick_insight(
    symbol: str,
    market: str = Query("us", description="Market: us, tw, cn, hk, crypto"),
):
    """
    快速洞察 - 获取股票的简要趋势分析

    适合快速浏览，返回各时间维度的信号方向和一句话总结。
    """
    from app.services.auto_analyzer import quick_analyze

    try:
        result = await quick_analyze(symbol=symbol, market=market)

        # 提取各时间维度信号
        analyses = result.get("timeframe_analyses", [])

        short_term = "neutral"
        mid_term = "neutral"
        long_term = "neutral"

        for a in analyses:
            tf = a.get("timeframe", "")
            signal = a.get("signal", "neutral")

            if "daily" in tf:
                short_term = signal
            elif "weekly" in tf:
                mid_term = signal
            elif "monthly" in tf:
                long_term = signal

        # 生成一句话总结
        overall = result.get("overall_signal", "neutral")
        confidence = result.get("overall_confidence", 0.5)

        signal_text = {
            "strong_bullish": "强势看涨",
            "bullish": "偏多",
            "neutral": "中性震荡",
            "bearish": "偏空",
            "strong_bearish": "强势看跌",
        }

        summary = f"{symbol} 当前{signal_text.get(overall, '走势不明')}，"

        if overall in ["strong_bullish", "bullish"]:
            summary += "历史相似形态后续表现积极。"
        elif overall in ["strong_bearish", "bearish"]:
            summary += "历史相似形态后续表现较弱，注意风险。"
        else:
            summary += "各周期信号存在分歧，建议观望。"

        return QuickInsightResponse(
            symbol=symbol,
            market=market,
            short_term_signal=short_term,
            mid_term_signal=mid_term,
            long_term_signal=long_term,
            overall_signal=overall,
            confidence=confidence,
            one_line_summary=summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
