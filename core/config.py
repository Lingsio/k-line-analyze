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
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "outputs" / "models"
    FAISS_INDEX_DIR: Path = BASE_DIR / "outputs" / "indices"

    # Model settings - RTX 4060 OPTIMIZED (8GB VRAM)
    MODEL_NAME: str = "KLine Pattern Encoder"
    EMBEDDING_DIM: int = 512  # Balanced for 8GB VRAM
    IMAGE_SIZE: int = 256  # Optimal resolution for 8GB
    DEFAULT_WINDOW_SIZES: List[int] = [10, 20, 30, 40, 60, 90, 120, 180, 240]  # Extended coverage
    
    # Search settings - 8GB OPTIMIZED
    DEFAULT_TOP_K: int = 50  # Good coverage
    FAISS_NPROBE: int = 512  # Balanced speed and precision
    # Data Fetcher Settings
    CACHE_EXPIRY_HOURS: int = 24

    # LLM API Settings (Alibaba Qwen / Google Gemini)
    LLM_PROVIDER: str = "gemini" # gemini or qwen
    LLM_API_KEY: str = ""  
    LLM_MODEL: str = "gemini-2.0-flash" 
    LLM_ENABLED: bool = False

    # LLM Cache Settings
    LLM_CACHE_DIR: Path = BASE_DIR.parent / "data" / "llm_cache"
    LLM_CACHE_EXPIRY_DAYS: int = 7

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
