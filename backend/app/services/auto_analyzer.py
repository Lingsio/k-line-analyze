"""
Auto Analyzer - 自动分析近期股票走势

功能：
1. 自动截取最近的日K、周K、月K
2. 对不同时间尺度进行相似度搜索
3. 综合多个时间维度给出分析报告
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class TimeFrame(str, Enum):
    """时间周期枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TimeFrameAnalysis:
    """单一时间周期的分析结果"""
    timeframe: str
    window_days: int
    start_date: str
    end_date: str
    similar_patterns: List[Dict[str, Any]]
    win_rate_5: float
    win_rate_20: float
    avg_return_5: float
    avg_return_20: float
    signal: str  # bullish, bearish, neutral
    confidence: float


@dataclass
class ComprehensiveAnalysis:
    """综合分析结果"""
    symbol: str
    market: str
    analysis_date: str
    timeframe_analyses: List[TimeFrameAnalysis]
    overall_signal: str
    overall_confidence: float
    recommendation: str
    risk_level: str  # low, medium, high
    key_insights: List[str]
    llm_analysis: Optional[str] = None  # LLM生成的分析报告


class AutoAnalyzer:
    """
    自动分析器 - 综合多时间维度分析股票走势
    """

    # 各时间周期的窗口配置
    TIMEFRAME_CONFIG = {
        TimeFrame.DAILY: {
            "windows": [20, 60],  # 分析最近20天和60天
            "period": "daily",
            "description": "短期趋势 (日K)",
        },
        TimeFrame.WEEKLY: {
            "windows": [12, 26],  # 分析最近12周和26周
            "period": "weekly",
            "description": "中期趋势 (周K)",
        },
        TimeFrame.MONTHLY: {
            "windows": [6, 12],  # 分析最近6个月和12个月
            "period": "monthly",
            "description": "长期趋势 (月K)",
        },
    }

    def __init__(self):
        from app.services.data_fetcher import DataFetcher
        from app.services.preprocessor import KLinePreprocessor
        from app.services.similarity_search import SimilaritySearchEngine
        from app.services.analyzer import PatternAnalyzer
        from app.services.feature_extractor import FeatureExtractor
        from app.config import settings

        self.data_fetcher = DataFetcher()
        self.preprocessor = KLinePreprocessor()
        self.search_engine = SimilaritySearchEngine()
        self.feature_extractor = FeatureExtractor()
        
        # 尝试加载索引
        if settings.FAISS_INDEX_DIR.exists():
            self.search_engine.load_index(settings.FAISS_INDEX_DIR)
            
        self.pattern_analyzer = PatternAnalyzer()

    # ============ LLM 缓存功能 ============
    
    def _get_cache_path(self, symbol: str, market: str) -> str:
        """获取缓存文件路径"""
        from app.config import settings
        import json
        
        # 确保缓存目录存在
        cache_dir = settings.LLM_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 symbol_market_date 作为文件名
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{symbol.upper()}_{market.lower()}_{today}.json"
        return cache_dir / filename
    
    def _load_cached_llm_analysis(self, symbol: str, market: str) -> Optional[Dict[str, Any]]:
        """
        尝试从缓存加载 LLM 分析结果
        
        Returns:
            缓存数据字典，包含 llm_analysis 和元数据；如果无效则返回 None
        """
        from app.config import settings
        import json
        
        cache_path = self._get_cache_path(symbol, market)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # 检查是否过期
            expires_at_str = cached_data.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    print(f"Cache expired for {symbol} ({market})")
                    return None
            
            print(f"Loaded LLM analysis from cache: {cache_path}")
            return cached_data
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None
    
    def _save_llm_analysis_to_cache(
        self, 
        symbol: str, 
        market: str, 
        llm_analysis: str,
        overall_signal: str = None,
        overall_confidence: float = None,
    ) -> bool:
        """
        将 LLM 分析结果保存到 JSON 缓存文件
        
        Returns:
            是否保存成功
        """
        from app.config import settings
        import json
        
        cache_path = self._get_cache_path(symbol, market)
        
        try:
            now = datetime.now()
            expires_at = now + timedelta(days=settings.LLM_CACHE_EXPIRY_DAYS)
            
            cache_data = {
                "symbol": symbol.upper(),
                "market": market.lower(),
                "analysis_date": now.strftime("%Y-%m-%d"),
                "created_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "llm_provider": settings.LLM_PROVIDER,
                "llm_model": settings.LLM_MODEL,
                "overall_signal": overall_signal,
                "overall_confidence": overall_confidence,
                "llm_analysis": llm_analysis,
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved LLM analysis to cache: {cache_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False
    
    def get_all_cached_analyses(self, symbol: str = None, market: str = None) -> List[Dict[str, Any]]:
        """
        获取所有缓存的分析列表（用于 API）
        
        Args:
            symbol: 可选，筛选特定股票
            market: 可选，筛选特定市场
            
        Returns:
            缓存文件摘要列表
        """
        from app.config import settings
        import json
        
        cache_dir = settings.LLM_CACHE_DIR
        if not cache_dir.exists():
            return []
            
        results = []
        for cache_file in cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 筛选
                if symbol and data.get("symbol", "").upper() != symbol.upper():
                    continue
                if market and data.get("market", "").lower() != market.lower():
                    continue
                
                # 返回摘要（不包含完整 llm_analysis 内容）
                results.append({
                    "symbol": data.get("symbol"),
                    "market": data.get("market"),
                    "analysis_date": data.get("analysis_date"),
                    "created_at": data.get("created_at"),
                    "expires_at": data.get("expires_at"),
                    "llm_provider": data.get("llm_provider"),
                    "llm_model": data.get("llm_model"),
                    "cache_file": cache_file.name,
                })
            except:
                continue
                
        return results
    
    def delete_cache(self, symbol: str, market: str) -> bool:
        """删除指定股票的缓存"""
        cache_path = self._get_cache_path(symbol, market)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Deleted cache: {cache_path}")
            return True
        return False

    async def analyze_stock(
        self,
        symbol: str,
        market: str = "us",
        timeframes: List[TimeFrame] = None,
        search_scope: List[str] = None,
        top_k: int = 10,
        self_only: bool = False,  # 仅搜索自身历史数据
        use_llm: bool = False,    # 是否使用LLM增强分析
        force_refresh: bool = False,  # 是否强制刷新（忽略缓存）
    ) -> ComprehensiveAnalysis:
        """
        综合分析股票的近期走势

        Args:
            symbol: 股票代码
            market: 市场
            timeframes: 要分析的时间周期，默认全部
            search_scope: 搜索范围
            top_k: 每个周期返回的相似结果数

        Returns:
            ComprehensiveAnalysis 综合分析结果
        """
        timeframes = timeframes or [TimeFrame.DAILY, TimeFrame.WEEKLY, TimeFrame.MONTHLY]
        search_scope = search_scope or [market]
        
        # 如果仅搜索自身历史，需要特殊处理
        self._self_only = self_only
        self._target_symbol = symbol if self_only else None

        timeframe_analyses = []

        for tf in timeframes:
            config = self.TIMEFRAME_CONFIG[tf]

            for window in config["windows"]:
                analysis = await self._analyze_timeframe(
                    symbol=symbol,
                    market=market,
                    period=config["period"],
                    window_size=window,
                    search_scope=search_scope,
                    top_k=top_k,
                    self_only=self_only,
                )
                if analysis:
                    timeframe_analyses.append(analysis)

        # 综合所有时间维度的分析
        overall_signal, overall_confidence = self._calculate_overall_signal(timeframe_analyses)
        recommendation = self._generate_recommendation(timeframe_analyses, overall_signal)
        risk_level = self._assess_risk_level(timeframe_analyses)
        key_insights = self._extract_key_insights(timeframe_analyses)
        
        # 如果启用LLM，进行增强分析（传入最近的K线数据用于图表生成）
        llm_analysis = None
        if use_llm:
            # 获取最近60天的日K数据用于生成图表
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                kline_df = await self.data_fetcher.fetch_ohlcv(
                    symbol=symbol,
                    market=market,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    period="daily",
                )
            except:
                kline_df = None
                
            llm_analysis = await self._get_llm_analysis(
                symbol=symbol,
                market=market,
                timeframe_analyses=timeframe_analyses,
                overall_signal=overall_signal,
                overall_confidence=overall_confidence,
                kline_df=kline_df,
                force_refresh=force_refresh,
            )

        return ComprehensiveAnalysis(
            symbol=symbol,
            market=market,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            timeframe_analyses=timeframe_analyses,
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            recommendation=llm_analysis if llm_analysis else recommendation,
            risk_level=risk_level,
            key_insights=key_insights,
            llm_analysis=llm_analysis,
        )

    async def _analyze_timeframe(
        self,
        symbol: str,
        market: str,
        period: str,
        window_size: int,
        search_scope: List[str],
        top_k: int,
        self_only: bool = False,
    ) -> Optional[TimeFrameAnalysis]:
        """分析单一时间周期"""
        try:
            # 计算需要获取的数据范围
            end_date = datetime.now()
            if period == "daily":
                start_date = end_date - timedelta(days=window_size + 30)
            elif period == "weekly":
                start_date = end_date - timedelta(weeks=window_size + 10)
            else:  # monthly
                start_date = end_date - timedelta(days=window_size * 35)

            # 获取数据
            df = await self.data_fetcher.fetch_ohlcv(
                symbol=symbol,
                market=market,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                period=period,
            )

            if df is None or len(df) < window_size:
                return None

            # 截取最近的窗口
            window_df = df.iloc[-window_size:]
            window_start = window_df.index[0]
            window_end = window_df.index[-1]

            # 格式化日期
            if hasattr(window_start, "strftime"):
                start_str = window_start.strftime("%Y-%m-%d")
                end_str = window_end.strftime("%Y-%m-%d")
            else:
                start_str = str(window_start)
                end_str = str(window_end)

            # 搜索相似形态 (使用模拟数据，实际需要连接到索引)
            similar_patterns = await self._search_similar(
                df=window_df,
                symbol=symbol,
                market=market,
                search_scope=search_scope,
                top_k=top_k,
                period=period,
                self_only=self_only,
            )

            # 动态补全后续收益数据 (如果索引中缺失)
            similar_patterns = await self._enrich_patterns_with_returns(similar_patterns, period)

            # 分析结果统计
            stats = self._calculate_stats(similar_patterns)

            # 确定信号
            signal = self._determine_signal(stats["win_rate_5"], stats["avg_return_5"])

            return TimeFrameAnalysis(
                timeframe=f"{period}_{window_size}",
                window_days=window_size,
                start_date=start_str,
                end_date=end_str,
                similar_patterns=similar_patterns,
                win_rate_5=stats["win_rate_5"],
                win_rate_20=stats["win_rate_20"],
                avg_return_5=stats["avg_return_5"],
                avg_return_20=stats["avg_return_20"],
                signal=signal,
                confidence=stats["confidence"],
            )

        except Exception as e:
            print(f"Error analyzing {period} timeframe: {e}")
            return None

    async def _search_similar(
        self,
        df: pd.DataFrame,
        symbol: str,
        market: str,
        search_scope: List[str],
        top_k: int,
        period: str,
        self_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似形态

        Args:
            df: 当前K线数据
            symbol: 当前股票代码
            market: 当前市场
            search_scope: 搜索范围
            top_k: 返回数量
            period: 周期
            self_only: 是否仅搜索自身历史
        """
        import random

        # 检查索引是否可用
        if self.search_engine.is_index_loaded():
            # 使用真实索引搜索
            # 使用 FeatureExtractor 直接从 DataFrame 提取特征
            if hasattr(self, 'feature_extractor'):
                try:
                    query_vector = self.feature_extractor.extract_from_dataframe(df)
                    # 确保 query_vector 是 1D array
                    if hasattr(query_vector, 'flatten'):
                        query_vector = query_vector.flatten()
                    
                    if self_only:
                        results = self.search_engine.search(query_vector, top_k=top_k * 5, markets=search_scope)
                        # 过滤只保留同一股票的结果 (忽略前缀差异)
                        def is_same_symbol(s1, s2):
                            if s1 == s2: return True
                            # Remove non-alphanumeric and compare?
                            # Simple approach for CN: check if one ends with other
                            c1 = ''.join(filter(str.isalnum, str(s1))).upper()
                            c2 = ''.join(filter(str.isalnum, str(s2))).upper()
                            # Specific fix for SH/SZ prefix
                            if c1.endswith(c2) or c2.endswith(c1):
                                # Ensure sufficient length overlap (e.g. > 4 digits) to avoid partial match false positives?
                                # Stock codes usually > 4 chars.
                                return True
                            return False

                        results = [r for r in results if is_same_symbol(r.get("symbol"), symbol)][:top_k]
                        return results
                    else:
                        return self.search_engine.search(query_vector, top_k=top_k, markets=search_scope)
                except Exception as e:
                    print(f"Feature extraction or search failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return []
            
            # 如果没有加载 feature_extractor
            print("Feature extractor not initialized")
            return []

    async def _enrich_patterns_with_returns(
        self, patterns: List[Dict[str, Any]], period: str
    ) -> List[Dict[str, Any]]:
        """
        动态补充模式的后续收益数据
        如果索引中没有保存 T+5/T+20 收益，则实时计算
        """
        if not patterns:
            return []
            
        # 检查是否已有收益数据
        if "subsequent_returns" in patterns[0]:
            return patterns

        enriched_patterns = []
        for p in patterns:
            try:
                # 克隆以避免修改原始引用（尽管这里可能是新对象）
                new_p = p.copy()
                
                # 如果已有，跳过
                if "subsequent_returns" in new_p and new_p["subsequent_returns"]:
                    enriched_patterns.append(new_p)
                    continue

                symbol = new_p.get("symbol")
                market = new_p.get("market", "us") # 默认为us如果未知
                end_date_str = new_p.get("end_date")
                
                if not symbol or not end_date_str:
                    enriched_patterns.append(new_p)
                    continue

                # 计算需要获取的时间范围 (T+30 用于覆盖交易日)
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                future_start = end_date
                # 对于日线 T+20 需要约30自然日；对于周线 T+5 需要5周~35天
                days_forward = 40 if period == "daily" else 150 # 粗略估算
                future_end = end_date + timedelta(days=days_forward)
                
                # 获取后续数据
                # TODO: 批量获取可能更高效，但这里逐个获取简单可靠
                df = await self.data_fetcher.fetch_ohlcv(
                    symbol=symbol,
                    market=market,
                    start_date=future_start.strftime("%Y-%m-%d"),
                    end_date=future_end.strftime("%Y-%m-%d"),
                    period=period
                )
                
                returns = {"t+5": None, "t+20": None}
                
                if df is not None and not df.empty:
                    # df index is date.
                    # Price at pattern end (T=0) roughly index 0 (or close to it)
                    # Wait, start_date passed to fetch was the end_date of pattern.
                    # So df[0] should be the day of pattern end or next day.
                    # We need return from T=0 (close price of pattern end).
                    
                    # Pattern end date price
                    # Ideally we have it in pattern metadata? No.
                    # We assume df.iloc[0] is roughly T=0 or T+1.
                    # Let's use df search.
                    
                    base_price = None
                    # Try to find price at end_date
                    if end_date in df.index:
                        base_price = df.loc[end_date]["close"]
                    else:
                        # Fallback to first available if close enough?
                        # Or pre-fetch slightly earlier?
                        # Assume df.iloc[0] is close enough to T=0 base
                        if len(df) > 0:
                            base_price = df.iloc[0]["close"]
                    
                    if base_price:
                        # Calculate T+5
                        if len(df) >= 6: # T+0 to T+5 needs 6 rows? Or 5 days after?
                            # iloc[5] is the 6th day (T+5 if daily)
                            # For weekly, iloc[5] is T+5 weeks.
                            p5 = df.iloc[5]["close"] if len(df) > 5 else df.iloc[-1]["close"]
                            returns["t+5"] = (p5 - base_price) / base_price
                        
                        # Calculate T+20
                        target_idx = 20
                        if len(df) > target_idx:
                            p20 = df.iloc[target_idx]["close"]
                            returns["t+20"] = (p20 - base_price) / base_price
                
                new_p["subsequent_returns"] = returns
                enriched_patterns.append(new_p)

            except Exception as e:
                # print(f"Failed to enrich pattern {p.get('symbol')}: {e}")
                new_p["subsequent_returns"] = {"t+5": None, "t+20": None}
                enriched_patterns.append(new_p)
                
        return enriched_patterns

    def _calculate_stats(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算统计指标"""
        if not patterns:
            return {
                "win_rate_5": 0.5,
                "win_rate_20": 0.5,
                "avg_return_5": 0,
                "avg_return_20": 0,
                "confidence": 0,
            }

        returns_5 = []
        returns_20 = []
        
        for p in patterns:
            sub = p.get("subsequent_returns", {})
            if sub and sub.get("t+5") is not None:
                returns_5.append(sub.get("t+5"))
            if sub and sub.get("t+20") is not None:
                returns_20.append(sub.get("t+20"))

        win_rate_5 = sum(1 for r in returns_5 if r > 0) / len(returns_5) if returns_5 else 0.5
        win_rate_20 = sum(1 for r in returns_20 if r > 0) / len(returns_20) if returns_20 else 0.5

        avg_return_5 = np.mean(returns_5) if returns_5 else 0
        avg_return_20 = np.mean(returns_20) if returns_20 else 0

        # 计算置信度
        sample_size = len(patterns)
        consistency = 1 - abs(win_rate_5 - win_rate_20)
        confidence = min(1.0, (np.log10(sample_size + 1) / 2) * 0.5 + consistency * 0.5)

        return {
            "win_rate_5": round(win_rate_5, 3),
            "win_rate_20": round(win_rate_20, 3),
            "avg_return_5": round(avg_return_5, 4),
            "avg_return_20": round(avg_return_20, 4),
            "confidence": round(confidence, 3),
        }

    def _determine_signal(self, win_rate: float, avg_return: float) -> str:
        """确定信号方向"""
        if win_rate > 0.6 and avg_return > 0.02:
            return "strong_bullish"
        elif win_rate > 0.55 or avg_return > 0.01:
            return "bullish"
        elif win_rate < 0.4 and avg_return < -0.02:
            return "strong_bearish"
        elif win_rate < 0.45 or avg_return < -0.01:
            return "bearish"
        else:
            return "neutral"

    def _calculate_overall_signal(
        self,
        analyses: List[TimeFrameAnalysis],
    ) -> tuple[str, float]:
        """综合计算整体信号"""
        if not analyses:
            return "neutral", 0.5

        # 信号权重：长期周期权重更高
        weights = {
            "daily": 0.3,
            "weekly": 0.4,
            "monthly": 0.3,
        }

        signal_scores = {
            "strong_bullish": 2,
            "bullish": 1,
            "neutral": 0,
            "bearish": -1,
            "strong_bearish": -2,
        }

        total_score = 0
        total_weight = 0
        confidences = []

        for analysis in analyses:
            period = analysis.timeframe.split("_")[0]
            weight = weights.get(period, 0.3)

            score = signal_scores.get(analysis.signal, 0)
            total_score += score * weight * analysis.confidence
            total_weight += weight
            confidences.append(analysis.confidence)

        if total_weight == 0:
            return "neutral", 0.5

        avg_score = total_score / total_weight
        avg_confidence = np.mean(confidences)

        # 转换分数为信号
        if avg_score > 1.2:
            signal = "strong_bullish"
        elif avg_score > 0.4:
            signal = "bullish"
        elif avg_score < -1.2:
            signal = "strong_bearish"
        elif avg_score < -0.4:
            signal = "bearish"
        else:
            signal = "neutral"

        return signal, round(avg_confidence, 3)

    def _generate_recommendation(
        self,
        analyses: List[TimeFrameAnalysis],
        overall_signal: str,
    ) -> str:
        """生成投资建议"""
        # 检查各时间维度一致性
        signals = [a.signal for a in analyses]
        bullish_count = sum(1 for s in signals if "bullish" in s)
        bearish_count = sum(1 for s in signals if "bearish" in s)

        if overall_signal == "strong_bullish":
            if bullish_count >= len(signals) * 0.8:
                return "多时间维度一致看涨，可考虑建仓或加仓。建议设置止损位。"
            else:
                return "整体偏多，但存在分歧。可小仓位试探，关注短期走势确认。"

        elif overall_signal == "bullish":
            return "中短期偏多，可关注回调后的买入机会。注意控制仓位。"

        elif overall_signal == "strong_bearish":
            if bearish_count >= len(signals) * 0.8:
                return "多时间维度一致看跌，建议规避或减仓。已持仓者注意止损。"
            else:
                return "整体偏空，但可能存在反弹机会。不建议追空，观望为主。"

        elif overall_signal == "bearish":
            return "短期偏弱，建议谨慎观望。等待企稳信号再做决策。"

        else:
            return "市场方向不明，各时间维度信号分歧。建议观望，等待更明确的趋势。"

    def _assess_risk_level(self, analyses: List[TimeFrameAnalysis]) -> str:
        """评估风险等级"""
        if not analyses:
            return "high"

        # 检查信号一致性
        signals = [a.signal for a in analyses]
        unique_signals = set(s.replace("strong_", "") for s in signals)

        # 检查置信度
        avg_confidence = np.mean([a.confidence for a in analyses])

        if len(unique_signals) == 1 and avg_confidence > 0.7:
            return "low"
        elif len(unique_signals) <= 2 and avg_confidence > 0.5:
            return "medium"
        else:
            return "high"

    def _extract_key_insights(self, analyses: List[TimeFrameAnalysis]) -> List[str]:
        """提取关键洞察"""
        insights = []

        if not analyses:
            return ["数据不足，无法生成分析"]

        # 按时间周期分组
        daily_analyses = [a for a in analyses if "daily" in a.timeframe]
        weekly_analyses = [a for a in analyses if "weekly" in a.timeframe]
        monthly_analyses = [a for a in analyses if "monthly" in a.timeframe]

        # 短期趋势
        if daily_analyses:
            avg_wr = np.mean([a.win_rate_5 for a in daily_analyses])
            if avg_wr > 0.6:
                insights.append(f"短期(日K)历史相似形态后续上涨概率较高 ({avg_wr*100:.0f}%)")
            elif avg_wr < 0.4:
                insights.append(f"短期(日K)历史相似形态后续下跌风险较大 (上涨概率仅{avg_wr*100:.0f}%)")

        # 中期趋势
        if weekly_analyses:
            avg_wr = np.mean([a.win_rate_5 for a in weekly_analyses])
            avg_ret = np.mean([a.avg_return_20 for a in weekly_analyses])
            if avg_wr > 0.55:
                insights.append(f"中期(周K)趋势偏多，历史平均收益 {avg_ret*100:+.1f}%")
            elif avg_wr < 0.45:
                insights.append(f"中期(周K)趋势偏空，需谨慎")

        # 长期趋势
        if monthly_analyses:
            avg_wr = np.mean([a.win_rate_20 for a in monthly_analyses])
            if avg_wr > 0.6:
                insights.append("长期(月K)形态健康，适合中长线布局")
            elif avg_wr < 0.4:
                insights.append("长期(月K)形态较弱，注意系统性风险")

        # 趋势一致性
        all_signals = [a.signal for a in analyses]
        if all("bullish" in s or s == "neutral" for s in all_signals):
            insights.append("多周期趋势方向一致，信号可靠性较高")
        elif all("bearish" in s or s == "neutral" for s in all_signals):
            insights.append("多周期趋势方向一致偏空")
        else:
            insights.append("不同周期信号存在分歧，建议观望")

        return insights[:5]  # 最多返回5条

    async def _get_llm_analysis(
        self,
        symbol: str,
        market: str,
        timeframe_analyses: List[TimeFrameAnalysis],
        overall_signal: str,
        overall_confidence: float,
        kline_df: pd.DataFrame = None,
        force_refresh: bool = False,
    ) -> Optional[str]:
        """
        使用 Gemini/Qwen 多模态进行智能分析（支持多张K线图）
        """
        from app.config import settings
        
        if not settings.LLM_ENABLED or not settings.LLM_API_KEY:
            return None
        
        # Check cache
        if not force_refresh:
            cached = self._load_cached_llm_analysis(symbol, market)
            if cached and cached.get("llm_analysis"):
                print(f"Using cached LLM analysis for {symbol} ({market})")
                return cached["llm_analysis"]
            
        try:
            # Prepare images list
            images = []
            image_descriptions = []
            
            # 1. Current stock chart
            if kline_df is not None and not kline_df.empty:
                current_img = self._generate_kline_chart(kline_df, symbol)
                if current_img:
                    images.append(current_img)
                    image_descriptions.append(f"Image 1: Current Stock {symbol} ({market}) Daily Chart")
            
            # 2. Historical similar patterns charts
            # Select top patterns (e.g., 1 from Daily, 1 from Weekly)
            historical_details = []
            processed_patterns = set()
            
            # Find candidate patterns
            candidates = []
            for tf in ["daily", "weekly"]:
                # Find analysis for this timeframe
                analysis = next((a for a in timeframe_analyses if tf in a.timeframe), None)
                if analysis and analysis.similar_patterns:
                    top_p = analysis.similar_patterns[0]
                    # Avoid duplicates
                    pid = f"{top_p.get('symbol')}_{top_p.get('end_date')}"
                    if pid not in processed_patterns:
                        candidates.append((tf, top_p))
                        processed_patterns.add(pid)
            
            # Fetch data and generate charts for candidates
            for i, (tf_name, pattern) in enumerate(candidates):
                try:
                    p_symbol = pattern.get("symbol")
                    p_market = pattern.get("market", "us")
                    p_end_date = datetime.strptime(pattern.get("end_date"), "%Y-%m-%d")
                    
                    # Fetch range: 60 days before end_date to 20 days after (to show outcome)
                    # Note: fetch_ohlcv expects strings
                    start_fetch = p_end_date - timedelta(days=90)
                    end_fetch = p_end_date + timedelta(days=30)
                    
                    hist_df = await self.data_fetcher.fetch_ohlcv(
                        symbol=p_symbol,
                        market=p_market,
                        start_date=start_fetch.strftime("%Y-%m-%d"),
                        end_date=end_fetch.strftime("%Y-%m-%d"),
                        period="daily" # Use daily for better visual comparison usually
                    )
                    
                    if hist_df is not None and not hist_df.empty:
                        hist_img = self._generate_kline_chart(hist_df, f"Similar: {p_symbol}")
                        if hist_img:
                            images.append(hist_img)
                            img_idx = len(images)
                            image_descriptions.append(f"Image {img_idx}: Historical Similar Pattern {p_symbol} ({p_market}) in {tf_name} timeframe. Pattern ended on {pattern.get('end_date')}.")
                            
                            # Update historical details text to reference image
                            historical_details.append(f"- Similar Pattern {img_idx-1} ({tf_name}): {p_symbol} on {pattern.get('end_date')}. See Image {img_idx}.")

                except Exception as e:
                    print(f"Failed to process historical pattern {pattern.get('symbol')}: {e}")
            
            # Build text details for all patterns (including those without charts)
            pattern_text_block = ""
            for a in timeframe_analyses:
                if a.similar_patterns:
                    top_patterns = a.similar_patterns[:3]
                    pattern_text_block += f"\n### {a.timeframe} Search Results:\n"
                    for i, p in enumerate(top_patterns, 1):
                        sub = p.get('subsequent_returns', {})
                        t5 = sub.get('t+5')
                        t20 = sub.get('t+20')
                        t5_str = f"{t5*100:+.2f}%" if t5 is not None else "N/A"
                        t20_str = f"{t20*100:+.2f}%" if t20 is not None else "N/A"
                        pattern_text_block += f"{i}. {p['symbol']} ({p['market']}) - Sim: {p['similarity_score']:.1%}, Returns: T+5={t5_str}, T+20={t20_str}\n"

            # Construct Prompt
            analysis_summary = []
            for a in timeframe_analyses:
                analysis_summary.append({
                    "timeframe": a.timeframe,
                    "signal": a.signal,
                    "win_rate_5d": f"{a.win_rate_5*100:.1f}%",
                    "win_rate_20d": f"{a.win_rate_20*100:.1f}%",
                    "avg_return_5d": f"{a.avg_return_5*100:+.2f}%",
                    "avg_return_20d": f"{a.avg_return_20*100:+.2f}%",
                    "confidence": f"{a.confidence*100:.0f}%",
                })

            prompt = f"""You are a senior Technical Analysis Expert and Quantitative Trader using Qwen-VL.
你是一位资深的技术分析专家和量化交易员。

**Input Images / 输入图片**:
{chr(10).join(image_descriptions)}

Please provide a comprehensive, professional, and in-depth technical analysis report for **{symbol}** ({market}).
请为股票 **{symbol}** ({market}市场) 提供一份专业、详尽的中英双语技术分析报告。

## 📊 Quantitative Analysis Data / 量化分析数据
**Overall Signal**: {overall_signal} (Confidence: {overall_confidence*100:.1f}%)

**Multi-Timeframe Stats**:
{self._format_analysis_for_llm(analysis_summary)}

## 📖 Historical Patterns Comparison / 历史形态对比
We have identified similar historical patterns from our database.
**Analysis of Similar Patterns**:
{pattern_text_block}

**Visual Comparison Task**:
- **Compare Image 1 (Current) with other images (Historical Patterns).**
- Analyze the similarity in trend, candlestick shapes, and volume.
- Discuss how the historical patterns evolved (look at the right side of historical charts).
- **Comparing** the current setup with historical outcomes is CRITICAL.

## 📝 Report Structure / 报告结构

### 1. Current Technical Analysis (Based on Image 1)
- Trend, Patterns, Key Levels.

### 2. Historical Similarity Analysis (Visual Comparison)
- Explicitly refence Image 2, Image 3 etc.
- "As seen in Image 2, stock XYZ showed a similar bottoming structure..."
- "Looking at the outcome in Image 2..."

### 3. Quantitative Insights
- Integrate the win rates and return stats provided above.

### 4. Conclusion & Strategy
- Practical trading advice.

## 📋 Output Requirements
1. **Bilingual**: English paragraph followed by Chinese paragraph.
2. **Professional**: Use institutional terminology.
3. **Deep**: Detailed visual analysis of provided charts.

Please start the analysis:"""

            result = None
            if settings.LLM_PROVIDER == "qwen":
                result = await self._generate_with_qwen(prompt, images)
            else:
                # Fallback/Legacy for Gemini (just use first image)
                result = await self._generate_with_gemini(prompt, images[0] if images else None)
            
            if result:
                self._save_llm_analysis_to_cache(
                    symbol=symbol, 
                    market=market, 
                    llm_analysis=result,
                    overall_signal=overall_signal,
                    overall_confidence=overall_confidence
                )
            
            return result
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _generate_with_gemini(self, prompt: str, image_base64: Optional[str]) -> Optional[str]:
        """Google Gemini API"""
        try:
            from app.config import settings
            import httpx
            
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.LLM_MODEL}:generateContent?key={settings.LLM_API_KEY}"
            
            parts = []
            if image_base64:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_base64
                    }
                })
            parts.append({"text": prompt})
            
            request_body = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 4096,
                }
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json=request_body
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        content = data["candidates"][0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return None
                else:
                    print(f"Gemini API error: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            print(f"Gemini generation failed: {e}")
            return None

    async def _generate_with_qwen(self, prompt: str, images: List[str]) -> Optional[str]:
        """
        Alibaba Qwen-VL API (DashScope)
        Supports multiple images
        """
        try:
            from app.config import settings
            import httpx
            
            api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
            
            messages = [
                {
                    "role": "system",
                    "content": [{"text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            user_content = messages[1]["content"]
            
            # Add images
            if images:
                for img_b64 in images:
                    if img_b64:
                        user_content.append({"image": f"data:image/png;base64,{img_b64}"})
                
            user_content.append({"text": prompt})
            
            request_body = {
                "model": settings.LLM_MODEL, # qwen-vl-max
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "top_p": 0.8,
                    "top_k": 50,
                    "max_tokens": 1500
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    api_url,
                    headers=headers,
                    json=request_body
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "output" in data and "text" in data["output"]:
                        return data["output"]["text"]
                    if "code" in data:
                        print(f"Qwen API logical error: {data}")
                    return None
                else:
                    print(f"Qwen API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"Qwen generation failed: {e}")
            return None
                    

    
    def _generate_kline_chart(self, df: pd.DataFrame, symbol: str) -> Optional[str]:
        """
        生成K线图并返回 base64 编码
        
        Args:
            df: K线数据
            symbol: 股票代码
            
        Returns:
            base64 编码的 PNG 图像
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from mplfinance.original_flavor import candlestick_ohlc
            from io import BytesIO
            import base64
            
            # 准备数据
            df_plot = df.copy()
            
            # 标准化列名
            col_map = {}
            for col in df_plot.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    col_map[col] = 'Open'
                elif 'high' in col_lower:
                    col_map[col] = 'High'
                elif 'low' in col_lower:
                    col_map[col] = 'Low'
                elif 'close' in col_lower:
                    col_map[col] = 'Close'
                elif 'volume' in col_lower:
                    col_map[col] = 'Volume'
            df_plot = df_plot.rename(columns=col_map)
            
            # 确保有日期索引
            if not isinstance(df_plot.index, pd.DatetimeIndex):
                df_plot.index = pd.to_datetime(df_plot.index)
            
            # 转换为 matplotlib 日期格式
            df_plot['Date'] = mdates.date2num(df_plot.index)
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           facecolor='#1a1a2e')
            
            ax1.set_facecolor('#1a1a2e')
            ax2.set_facecolor('#1a1a2e')
            
            # 绘制K线
            ohlc_data = df_plot[['Date', 'Open', 'High', 'Low', 'Close']].values
            candlestick_ohlc(ax1, ohlc_data, width=0.6, 
                           colorup='#26a69a', colordown='#ef5350',
                           alpha=0.9)
            
            # 添加均线
            if len(df_plot) >= 5:
                ma5 = df_plot['Close'].rolling(5).mean()
                ax1.plot(df_plot['Date'], ma5, color='#ffd700', linewidth=1, label='MA5')
            if len(df_plot) >= 20:
                ma20 = df_plot['Close'].rolling(20).mean()
                ax1.plot(df_plot['Date'], ma20, color='#00bcd4', linewidth=1, label='MA20')
            
            # 设置标题和样式
            ax1.set_title(f'{symbol} K-Line Chart', color='white', fontsize=14, pad=10)
            ax1.set_ylabel('Price', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white')
            ax1.spines['bottom'].set_color('#333')
            ax1.spines['top'].set_color('#333')
            ax1.spines['left'].set_color('#333')
            ax1.spines['right'].set_color('#333')
            
            # 绘制成交量
            if 'Volume' in df_plot.columns:
                colors = ['#26a69a' if df_plot['Close'].iloc[i] >= df_plot['Open'].iloc[i] 
                         else '#ef5350' for i in range(len(df_plot))]
                ax2.bar(df_plot['Date'], df_plot['Volume'], color=colors, alpha=0.7, width=0.6)
                ax2.set_ylabel('Volume', color='white')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.2)
                ax2.spines['bottom'].set_color('#333')
                ax2.spines['top'].set_color('#333')
                ax2.spines['left'].set_color('#333')
                ax2.spines['right'].set_color('#333')
            
            # 格式化x轴日期
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            
            plt.tight_layout()
            
            # 保存到内存
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, facecolor='#1a1a2e', 
                       edgecolor='none', bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            
            # 转为 base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return image_base64
            
        except Exception as e:
            print(f"Failed to generate K-line chart: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_analysis_for_llm(self, analyses: List[Dict]) -> str:
        """格式化分析数据供LLM阅读"""
        lines = []
        for a in analyses:
            lines.append(f"- {a['timeframe']}: 信号={a['signal']}, 5日胜率={a['win_rate_5d']}, 20日胜率={a['win_rate_20d']}, 5日均收益={a['avg_return_5d']}, 样本数={a['similar_count']}, 置信度={a['confidence']}")
        return "\n".join(lines)


async def quick_analyze(symbol: str, market: str = "us", self_only: bool = False, use_llm: bool = False, force_refresh: bool = False) -> Dict[str, Any]:
    """
    快速分析接口 - 供 API 调用

    Args:
        symbol: 股票代码
        market: 市场
        self_only: 仅搜索自身历史
        use_llm: 使用LLM增强分析
        force_refresh: 是否强制刷新（忽略缓存）

    Returns:
        分析结果字典
    """
    analyzer = AutoAnalyzer()
    
    # 根据self_only决定搜索范围
    search_scope = [market] if self_only else [market, "us"]
    
    result = await analyzer.analyze_stock(
        symbol=symbol,
        market=market,
        search_scope=search_scope,
        self_only=self_only,
        use_llm=use_llm,
        force_refresh=force_refresh,
    )

    # 转换为字典格式
    return {
        "symbol": result.symbol,
        "market": result.market,
        "analysis_date": result.analysis_date,
        "overall_signal": result.overall_signal,
        "overall_confidence": result.overall_confidence,
        "recommendation": result.recommendation,
        "risk_level": result.risk_level,
        "key_insights": result.key_insights,
        "llm_analysis": result.llm_analysis,
        "timeframe_analyses": [
            {
                "timeframe": a.timeframe,
                "window_days": a.window_days,
                "period": a.start_date + " ~ " + a.end_date,
                "start_date": a.start_date,
                "end_date": a.end_date,
                "signal": a.signal,
                "win_rate_5d": a.win_rate_5,
                "win_rate_20d": a.win_rate_20,
                "avg_return_5d": a.avg_return_5,
                "avg_return_20d": a.avg_return_20,
                "confidence": a.confidence,
                "similar_count": len(a.similar_patterns),
                "similar_patterns": a.similar_patterns,
            }
            for a in result.timeframe_analyses
        ],
    }
