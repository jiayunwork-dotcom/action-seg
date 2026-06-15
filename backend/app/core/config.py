import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "Action Segmentation API"
    DEBUG: bool = True

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    RESULTS_DIR: Path = STORAGE_DIR / "results"

    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: set = {"mp4", "avi", "mov"}

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

    CELERY_WORKER_CONCURRENCY: int = 3

    DATA_EXPIRY_DAYS: int = 7

    CONFIGS_DIR: Path = Path(os.getenv("CONFIGS_DIR", str(BASE_DIR.parent / "configs")))
    ACTION_CLASSES_PATH: Path = CONFIGS_DIR / "action_classes.json"
    MODEL_CONFIG_PATH: Path = CONFIGS_DIR / "model_config.json"

    RANDOM_SEED: int = 42

    class Config:
        env_file = ".env"


settings = Settings()

for d in [settings.STORAGE_DIR, settings.UPLOAD_DIR, settings.CACHE_DIR, settings.RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
