"""
Prediction API routes for CNN and Transformer stock predictors.

Provides endpoints for:
- Single stock prediction using trained models
- Model info and available predictors listing
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import torch
import json

from app.config import settings

router = APIRouter(prefix="/prediction", tags=["Prediction"])


class PredictionRequest(BaseModel):
    symbol: str
    market: str = "us"
    model_type: str = "transformer"  # "transformer" or "cnn"
    model_variant: str = "universal"  # "universal" or stock-specific name
    window_size: int = 60


class PredictionResult(BaseModel):
    symbol: str
    market: str
    model_type: str
    predictions: Dict  # {horizon: {class_name, probability, predicted_return}}


@router.get("/models")
async def list_available_models():
    """List all available prediction models."""
    models_dir = settings.MODELS_DIR / "predictors"
    if not models_dir.exists():
        return {"models": []}

    models = []
    for f in models_dir.glob("*.pt"):
        name = f.stem
        config_path = models_dir / f"{name}_config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as cf:
                config = json.load(cf)

        model_type = "transformer" if "transformer" in name else "cnn"
        variant = "universal" if "universal" in name else name

        models.append({
            "name": name,
            "type": model_type,
            "variant": variant,
            "file": str(f),
            "config": config,
        })

    return {"models": models}


@router.post("/predict", response_model=PredictionResult)
async def predict_stock(request: PredictionRequest):
    """
    Predict future price movement for a stock.

    Uses the specified model type (CNN or Transformer) and variant
    (universal or stock-specific) to make predictions.
    """
    from app.services.data_fetcher import DataFetcher
    from datetime import datetime, timedelta

    # Determine model path
    models_dir = settings.MODELS_DIR / "predictors"
    if request.model_variant == "universal":
        model_file = f"{request.model_type}_predictor_universal.pt"
    else:
        model_file = f"{request.model_type}_predictor_{request.market}_{request.symbol}.pt"

    model_path = models_dir / model_file
    if not model_path.exists():
        # Fallback to universal
        model_file = f"{request.model_type}_predictor_universal.pt"
        model_path = models_dir / model_file
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found: {model_file}. "
                       f"Train a model first with scripts/train_{request.model_type}_predictor.py",
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if request.model_type == "transformer":
        from app.models.transformer_predictor import KLineTransformerPredictor
        model = KLineTransformerPredictor.load(str(model_path), device=device)
    else:
        from app.models.cnn_predictor import KLineCNNPredictor
        model = KLineCNNPredictor.load(str(model_path), device=device)

    model = model.to(device)
    model.eval()

    # Fetch recent data
    fetcher = DataFetcher()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=request.window_size * 3)).strftime(
        "%Y-%m-%d"
    )

    df = await fetcher.fetch_ohlcv(
        symbol=request.symbol,
        market=request.market,
        start_date=start_date,
        end_date=end_date,
    )

    if df is None or len(df) < request.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {request.symbol}. "
                   f"Need {request.window_size} bars, got {len(df) if df is not None else 0}",
        )

    # Use most recent window
    window_df = df.iloc[-request.window_size:]

    # Prepare input based on model type
    if request.model_type == "transformer":
        from app.models.prediction_dataset import StockPredictionDataset
        temp_ds = StockPredictionDataset.__new__(StockPredictionDataset)
        sequence = temp_ds._df_to_sequence(window_df)
        input_tensor = sequence.unsqueeze(0).to(device)
        results = model.predict(input_tensor)
    else:
        from app.models.cnn_encoder import create_default_transforms
        from app.services.preprocessor import KLinePreprocessor

        _, inf_transform = create_default_transforms()
        preprocessor = KLinePreprocessor()
        normalized = preprocessor.normalize(window_df)
        image = preprocessor.to_kline_image(normalized, image_size=128)
        image_tensor = inf_transform(image).unsqueeze(0).to(device)
        results = model.predict(image_tensor)

    # Format predictions
    predictions = {}
    for horizon_key, result in results.items():
        predictions[horizon_key] = {
            "predicted_class": result["predicted_class_name"][0],
            "class_probabilities": {
                name: float(prob)
                for name, prob in zip(
                    model.CLASS_NAMES, result["class_probs"][0]
                )
            },
            "predicted_return": float(result["predicted_return"][0]),
        }

    return PredictionResult(
        symbol=request.symbol,
        market=request.market,
        model_type=request.model_type,
        predictions=predictions,
    )


@router.get("/compare/{symbol}")
async def compare_models(
    symbol: str,
    market: str = Query(default="us"),
    window_size: int = Query(default=60),
):
    """
    Compare predictions from all available models for a given stock.

    Returns predictions from both CNN and Transformer models
    (universal + stock-specific if available).
    """
    models_dir = settings.MODELS_DIR / "predictors"
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="No trained models found")

    results = {}
    for model_file in models_dir.glob("*.pt"):
        name = model_file.stem
        model_type = "transformer" if "transformer" in name else "cnn"

        try:
            request = PredictionRequest(
                symbol=symbol,
                market=market,
                model_type=model_type,
                model_variant=name,
                window_size=window_size,
            )
            prediction = await predict_stock(request)
            results[name] = prediction.predictions
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "symbol": symbol,
        "market": market,
        "model_comparisons": results,
    }
