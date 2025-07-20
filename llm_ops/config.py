from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    weaviate_url: str = "http://localhost:8080"
    embed_model: str = "intfloat/e5-base-v2"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
    mlflow_tracking_uri: str = "file:./mlruns"   

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore unrelated env vars
    )


@lru_cache
def get_settings() -> Settings:  # pragma: no cover
    return Settings()
