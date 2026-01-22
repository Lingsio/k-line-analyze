from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date

router = APIRouter()


class SearchRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to search")
    market: str = Field("us", description="Market: us, tw, cn, hk, crypto")
    start_date: str = Field(..., description="Start date of K-line pattern (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date of K-line pattern (YYYY-MM-DD)")
    window_size: int = Field(30, ge=10, le=120, description="Window size in days")
    top_k: int = Field(10, ge=1, le=100, description="Number of similar patterns to return")
    search_scope: List[str] = Field(
        ["us"], description="Markets to search in"
    )
    include_volume: bool = Field(True, description="Include volume in similarity matching")
    use_dtw_refinement: bool = Field(True, description="Use DTW for final ranking refinement")


class SubsequentReturns(BaseModel):
    t_plus_1: Optional[float] = Field(None, description="Return after 1 day")
    t_plus_5: Optional[float] = Field(None, description="Return after 5 days")
    t_plus_10: Optional[float] = Field(None, description="Return after 10 days")
    t_plus_20: Optional[float] = Field(None, description="Return after 20 days")


class SimilarPattern(BaseModel):
    symbol: str
    market: str
    start_date: str
    end_date: str
    similarity_score: float = Field(..., ge=0, le=1)
    dtw_distance: Optional[float] = None
    subsequent_returns: SubsequentReturns
    kline_data: Optional[List[dict]] = None


class AnalysisSummary(BaseModel):
    total_matches: int
    win_rate_1d: float = Field(..., description="Win rate for T+1")
    win_rate_5d: float = Field(..., description="Win rate for T+5")
    win_rate_20d: float = Field(..., description="Win rate for T+20")
    avg_return_1d: float
    avg_return_5d: float
    avg_return_20d: float
    median_return_5d: float
    confidence: float = Field(..., ge=0, le=1)


class QueryInfo(BaseModel):
    symbol: str
    market: str
    start_date: str
    end_date: str
    window_size: int
    normalized_close: Optional[List[float]] = None


class SearchResponse(BaseModel):
    query_info: QueryInfo
    similar_patterns: List[SimilarPattern]
    analysis: AnalysisSummary


@router.post("/search/similar", response_model=SearchResponse)
async def search_similar_patterns(request: SearchRequest):
    """
    Search for similar K-line patterns in historical data.

    This endpoint:
    1. Fetches the query K-line pattern
    2. Normalizes and encodes it using the CNN model
    3. Searches FAISS index for similar vectors
    4. Optionally refines results using DTW
    5. Analyzes subsequent returns of matched patterns
    """
    from app.services.data_fetcher import DataFetcher
    from app.services.preprocessor import KLinePreprocessor
    from app.services.similarity_search import SimilaritySearchEngine
    from app.services.dtw_matcher import DTWMatcher
    from app.services.analyzer import PatternAnalyzer

    try:
        # Initialize services
        data_fetcher = DataFetcher()
        preprocessor = KLinePreprocessor()
        search_engine = SimilaritySearchEngine()
        dtw_matcher = DTWMatcher()
        analyzer = PatternAnalyzer()

        # 1. Fetch query K-line data
        query_df = await data_fetcher.fetch_ohlcv(
            symbol=request.symbol,
            market=request.market,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        if query_df is None or len(query_df) < 10:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for the selected date range",
            )

        # 2. Preprocess and normalize
        normalized_data = preprocessor.normalize(query_df)
        query_vector = preprocessor.to_feature_vector(normalized_data)

        # 3. Search FAISS index
        if not search_engine.is_index_loaded():
            # If no index is built yet, return mock data for demo
            return _create_mock_response(request)

        candidates = search_engine.search(
            query_vector=query_vector,
            top_k=request.top_k * 3,  # Get more for DTW refinement
            markets=request.search_scope,
        )

        # 4. DTW refinement (optional)
        if request.use_dtw_refinement and candidates:
            candidates = dtw_matcher.refine_matches(
                query_series=normalized_data["close"].values,
                candidates=candidates,
                top_k=request.top_k,
            )
        else:
            candidates = candidates[: request.top_k]

        # 5. Fetch subsequent returns and build response
        similar_patterns = []
        for candidate in candidates:
            returns = await analyzer.get_subsequent_returns(
                symbol=candidate["symbol"],
                market=candidate["market"],
                end_date=candidate["end_date"],
            )

            similar_patterns.append(
                SimilarPattern(
                    symbol=candidate["symbol"],
                    market=candidate["market"],
                    start_date=candidate["start_date"],
                    end_date=candidate["end_date"],
                    similarity_score=candidate["similarity_score"],
                    dtw_distance=candidate.get("dtw_distance"),
                    subsequent_returns=SubsequentReturns(
                        t_plus_1=returns.get("t+1"),
                        t_plus_5=returns.get("t+5"),
                        t_plus_10=returns.get("t+10"),
                        t_plus_20=returns.get("t+20"),
                    ),
                )
            )

        # 6. Generate analysis summary
        analysis = analyzer.analyze_outcomes(similar_patterns)

        return SearchResponse(
            query_info=QueryInfo(
                symbol=request.symbol,
                market=request.market,
                start_date=request.start_date,
                end_date=request.end_date,
                window_size=request.window_size,
                normalized_close=normalized_data["close"].tolist()
                if normalized_data is not None
                else None,
            ),
            similar_patterns=similar_patterns,
            analysis=analysis,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _create_mock_response(request: SearchRequest) -> SearchResponse:
    """Create a mock response for demo purposes when no index is available."""
    import random

    mock_patterns = []
    symbols = ["NVDA", "TSLA", "AMZN", "GOOGL", "META", "MSFT", "AMD", "NFLX"]
    years = ["2020", "2019", "2018", "2021", "2022"]

    for i in range(min(request.top_k, 8)):
        similarity = round(0.95 - i * 0.05 + random.uniform(-0.02, 0.02), 3)
        mock_patterns.append(
            SimilarPattern(
                symbol=symbols[i % len(symbols)],
                market="us",
                start_date=f"{years[i % len(years)]}-{random.randint(1,12):02d}-01",
                end_date=f"{years[i % len(years)]}-{random.randint(1,12):02d}-{random.randint(15,28):02d}",
                similarity_score=max(0.5, similarity),
                dtw_distance=round(random.uniform(0.1, 0.5), 3),
                subsequent_returns=SubsequentReturns(
                    t_plus_1=round(random.uniform(-0.03, 0.05), 4),
                    t_plus_5=round(random.uniform(-0.05, 0.10), 4),
                    t_plus_10=round(random.uniform(-0.08, 0.15), 4),
                    t_plus_20=round(random.uniform(-0.10, 0.20), 4),
                ),
            )
        )

    # Calculate mock analysis
    returns_5d = [p.subsequent_returns.t_plus_5 or 0 for p in mock_patterns]
    returns_1d = [p.subsequent_returns.t_plus_1 or 0 for p in mock_patterns]
    returns_20d = [p.subsequent_returns.t_plus_20 or 0 for p in mock_patterns]

    win_1d = sum(1 for r in returns_1d if r > 0) / len(returns_1d) if returns_1d else 0
    win_5d = sum(1 for r in returns_5d if r > 0) / len(returns_5d) if returns_5d else 0
    win_20d = sum(1 for r in returns_20d if r > 0) / len(returns_20d) if returns_20d else 0

    return SearchResponse(
        query_info=QueryInfo(
            symbol=request.symbol,
            market=request.market,
            start_date=request.start_date,
            end_date=request.end_date,
            window_size=request.window_size,
        ),
        similar_patterns=mock_patterns,
        analysis=AnalysisSummary(
            total_matches=len(mock_patterns),
            win_rate_1d=round(win_1d, 3),
            win_rate_5d=round(win_5d, 3),
            win_rate_20d=round(win_20d, 3),
            avg_return_1d=round(sum(returns_1d) / len(returns_1d), 4) if returns_1d else 0,
            avg_return_5d=round(sum(returns_5d) / len(returns_5d), 4) if returns_5d else 0,
            avg_return_20d=round(sum(returns_20d) / len(returns_20d), 4) if returns_20d else 0,
            median_return_5d=round(sorted(returns_5d)[len(returns_5d) // 2], 4) if returns_5d else 0,
            confidence=round(0.7 + random.uniform(0, 0.2), 3),
        ),
    )


@router.get("/search/status")
async def get_search_status():
    """Get status of the search engine (index loaded, etc.)."""
    from app.services.similarity_search import SimilaritySearchEngine

    engine = SimilaritySearchEngine()
    return {
        "index_loaded": engine.is_index_loaded(),
        "total_vectors": engine.get_total_vectors(),
        "markets_indexed": engine.get_indexed_markets(),
    }
