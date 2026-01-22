from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "K-Line Pattern Finder"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR.parent / "data"
    MODELS_DIR: Path = BASE_DIR / "trained_models"
    FAISS_INDEX_DIR: Path = BASE_DIR / "faiss_index"

    # Model Settings
    EMBEDDING_DIM: int = 256
    IMAGE_SIZE: int = 128
    DEFAULT_WINDOW_SIZES: List[int] = [20, 60, 120]

    # Search Settings
    DEFAULT_TOP_K: int = 10
    FAISS_NPROBE: int = 128

    # Data Fetcher Settings
    CACHE_EXPIRY_HOURS: int = 24

    # LLM API Settings (Gemini 多模态)
    LLM_API_KEY: str = ""  # Gemini API Key
    LLM_MODEL: str = "gemini-2.0-flash"  # gemini-2.0-flash 或 gemini-1.5-pro
    LLM_ENABLED: bool = False  # 设为 True 启用 LLM 分析

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
