"""Domain models shared across the application."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class Side(str, enum.Enum):
    YES = "YES"
    NO = "NO"


class OrderAction(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class MarketStatus(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


class Market(BaseModel):
    """Normalised representation of a prediction market."""

    id: str
    source: str
    question: str
    url: str = ""
    status: MarketStatus = MarketStatus.OPEN
    probability: float = Field(ge=0.0, le=1.0)
    volume_24h: float = 0.0
    liquidity: float = 0.0
    close_time: datetime | None = None
    created_at: datetime | None = None
    last_updated: datetime | None = None


class PricePoint(BaseModel):
    """A single historical price observation."""

    timestamp: datetime
    probability: float = Field(ge=0.0, le=1.0)
    volume: float = 0.0


class Signal(BaseModel):
    """Output of a signal generator."""

    name: str
    market_id: str
    value: float = Field(ge=-1.0, le=1.0, description="Ranges from -1 (strong NO) to +1 (strong YES)")
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, float] = Field(default_factory=dict)


class Order(BaseModel):
    """An order to place on a prediction market."""

    market_id: str
    source: str
    side: Side
    action: OrderAction
    amount: float = Field(gt=0.0)
    limit_price: float | None = Field(default=None, ge=0.0, le=1.0)


class Position(BaseModel):
    """Current position in a market."""

    market_id: str
    source: str
    side: Side
    shares: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_price)


class TradeDecision(BaseModel):
    """A strategy's recommendation to trade."""

    market_id: str
    source: str
    side: Side
    action: OrderAction
    target_amount: float = Field(gt=0.0)
    limit_price: float | None = None
    signals: list[Signal] = Field(default_factory=list)
    reason: str = ""
