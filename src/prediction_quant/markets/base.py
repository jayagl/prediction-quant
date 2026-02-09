"""Abstract base class for prediction market API clients."""

from __future__ import annotations

import abc
from typing import Sequence

from prediction_quant.models import Market, Order, Position, PricePoint


class MarketClient(abc.ABC):
    """Interface every market adapter must implement."""

    source: str  # e.g. "polymarket", "manifold"

    @abc.abstractmethod
    async def list_markets(
        self,
        *,
        limit: int = 50,
        active_only: bool = True,
        min_volume: float = 0.0,
    ) -> list[Market]:
        """Return available markets, optionally filtered."""

    @abc.abstractmethod
    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ID."""

    @abc.abstractmethod
    async def get_price_history(
        self,
        market_id: str,
        *,
        points: int = 100,
    ) -> list[PricePoint]:
        """Return recent price/probability history for a market."""

    @abc.abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return current open positions for the authenticated account."""

    @abc.abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place an order. Returns an order/trade ID."""

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""

    async def get_markets_batch(self, market_ids: Sequence[str]) -> list[Market]:
        """Fetch multiple markets. Override for APIs that support batch fetches."""
        results = []
        for mid in market_ids:
            results.append(await self.get_market(mid))
        return results

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources (HTTP clients, etc.)."""

    async def __aenter__(self) -> MarketClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
