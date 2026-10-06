"""Explicit configuration; constructing settings never makes network calls."""

from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql://risk:localdev@127.0.0.1:55432/riskqa"
    data_dir: Path = Path("data")
    openai_api_key: SecretStr | None = None
    openai_model: str = ""
    openai_input_usd_per_million: Decimal | None = None
    openai_output_usd_per_million: Decimal | None = None
    max_run_usd: Decimal = Field(default=Decimal("0"), ge=0)
    max_provider_calls: int = Field(default=8, ge=1, le=8)
    request_timeout_seconds: float = Field(default=120, gt=0)
    provider_timeout_seconds: float = Field(default=30, gt=0)
    max_output_tokens: int = Field(default=1500, gt=0)
    model_manifest: Path = Path("models/manifest.json")
    corpus_manifest: Path = Path("corpus/manifest.json")
