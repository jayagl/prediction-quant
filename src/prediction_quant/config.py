"""Configuration management via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class MarketSettings(BaseSettings):
    """API credentials and endpoints for prediction markets."""

    model_config = {"env_prefix": ""}

    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    manifold_api_key: str = ""


class BotSettings(BaseSettings):
    """Bot behaviour knobs."""

    model_config = {"env_prefix": "BOT_"}

    dry_run: bool = True
    poll_interval_seconds: int = 60
    max_position_size: float = 100.0
    max_total_exposure: float = 1000.0
    max_drawdown_pct: float = 0.10


def load_market_settings() -> MarketSettings:
    return MarketSettings(_env_file=".env", _env_file_encoding="utf-8")


def load_bot_settings() -> BotSettings:
    return BotSettings(_env_file=".env", _env_file_encoding="utf-8")
