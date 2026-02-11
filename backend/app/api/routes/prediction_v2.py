"""
Prediction API v2 routes — uses 10-day/256x256 models.

Endpoints:
- GET  /prediction/v2/models       — list trained v2 models
- POST /prediction/v2/predict      — predict with v2 model
- GET  /prediction/v2/compare/{s}  — compare all v2 models
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict
import torch
import json

from app.config import settings

router = APIRouter(prefix="/prediction/v2", tags=["Prediction V2"])

V2_IMAGE_SIZE = 256
V2_WINDOW_SIZE = 10


class PredictionRequestV2(BaseModel):
    symbol: str
    market: str = "us"
    model_type: str = "transformer"  # "transformer" or "cnn"
    model_variant: str = "universal"
    window_size: int = V2_WINDOW_SIZE


class PredictionResultV2(BaseModel):
    symbol: str
    market: str
    model_type: str
    window_size: int
    image_size: int
    predictions: Dict


@router.get("/models")
async def list_v2_models():
    """List all trained v2 prediction models."""
    models_dir = settings.MODELS_DIR / "predictors"
    if not models_dir.exists():
        return {"models": []}

    models = []
    for f in models_dir.glob("*_v2_*.pt"):
        name = f.stem
        config_path = models_dir / f"{name}_config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as cf:
                config = json.load(cf)

        model_type = "transformer" if "transformer" in name else "cnn"
        models.append({
            "name": name,
            "type": model_type,
            "variant": "universal" if "universal" in name else name,
            "file": str(f),
            "config": config,
        })

    return {"models": models}


@router.post("/predict", response_model=PredictionResultV2)
async def predict_stock_v2(request: PredictionRequestV2):
    """Predict using v2 models (10-day / 256x256)."""
    from app.services.data_fetcher import DataFetcher
    from datetime import datetime, timedelta

    models_dir = settings.MODELS_DIR / "predictors"

    # Resolve model file
    if request.model_type == "cnn":
        prefix = "cnn_v2"
    else:
        prefix = "transformer_v2"

    if request.model_variant == "universal":
        model_file = f"{prefix}_universal.pt"
    else:
        model_file = f"{prefix}_{request.market}_{request.symbol}.pt"

    model_path = models_dir / model_file
    if not model_path.exists():
        # Fallback to universal
        model_path = models_dir / f"{prefix}_universal.pt"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No v2 model found: {model_file}. "
                       f"Train with: scripts/train_{prefix.replace('_v2','')}_predictor_v2.py",
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if request.model_type == "transformer":
        from app.models.transformer_predictor import KLineTransformerPredictor
        model = KLineTransformerPredictor.load(str(model_path), device=device)
    else:
        from app.models.cnn_predictor_v2 import KLineCNNPredictorV2
        model = KLineCNNPredictorV2.load(str(model_path), device=device)

    model = model.to(device)
    model.eval()

    # Fetch recent data
    fetcher = DataFetcher()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=request.window_size * 5)).strftime("%Y-%m-%d")

    df = await fetcher.fetch_ohlcv(
        symbol=request.symbol, market=request.market,
        start_date=start_date, end_date=end_date,
    )

    if df is None or len(df) < request.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Need {request.window_size} bars, got {len(df) if df is not None else 0}",
        )

    window_df = df.iloc[-request.window_size:]

    # Prepare input
    if request.model_type == "transformer":
        from app.models.prediction_dataset_v2 import StockPredictionDatasetV2
        temp_ds = StockPredictionDatasetV2.__new__(StockPredictionDatasetV2)
        sequence = temp_ds._df_to_sequence(window_df)
        input_tensor = sequence.unsqueeze(0).to(device)
        results = model.predict(input_tensor)
    else:
        from app.models.cnn_encoder import create_default_transforms
        from app.services.preprocessor import KLinePreprocessor

        _, inf_transform = create_default_transforms()
        preprocessor = KLinePreprocessor()
        normalized = preprocessor.normalize(window_df)
        image = preprocessor.to_kline_image(normalized, image_size=V2_IMAGE_SIZE)
        image_tensor = inf_transform(image).unsqueeze(0).to(device)
        results = model.predict(image_tensor)

    # Format
    predictions = {}
    for horizon_key, result in results.items():
        predictions[horizon_key] = {
            "predicted_class": result["predicted_class_name"][0],
            "class_probabilities": {
                name: float(prob)
                for name, prob in zip(model.CLASS_NAMES, result["class_probs"][0])
            },
            "predicted_return": float(result["predicted_return"][0]),
        }

    return PredictionResultV2(
        symbol=request.symbol,
        market=request.market,
        model_type=request.model_type,
        window_size=request.window_size,
        image_size=V2_IMAGE_SIZE,
        predictions=predictions,
    )


@router.get("/compare/{symbol}")
async def compare_v2_models(
    symbol: str,
    market: str = Query(default="us"),
):
    """Compare all v2 models for a stock."""
    models_dir = settings.MODELS_DIR / "predictors"
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="No trained models")

    results = {}
    for model_file in models_dir.glob("*_v2_*.pt"):
        name = model_file.stem
        model_type = "transformer" if "transformer" in name else "cnn"

        try:
            req = PredictionRequestV2(
                symbol=symbol, market=market,
                model_type=model_type, model_variant=name,
            )
            pred = await predict_stock_v2(req)
            results[name] = pred.predictions
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"symbol": symbol, "market": market, "comparisons": results}
