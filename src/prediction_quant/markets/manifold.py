"""Manifold Markets API client."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from prediction_quant.markets.base import MarketClient
from prediction_quant.models import (
    Market,
    MarketStatus,
    Order,
    OrderAction,
    Position,
    PricePoint,
    Side,
)

_BASE_URL = "https://api.manifold.markets/v0"


class ManifoldClient(MarketClient):
    """Client for the Manifold Markets API."""

    source = "manifold"

    def __init__(self, api_key: str = "") -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Key {api_key}"
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    async def list_markets(
        self,
        *,
        limit: int = 50,
        active_only: bool = True,
        min_volume: float = 0.0,
    ) -> list[Market]:
        params: dict[str, str | int] = {
            "limit": limit,
            "sort": "24-hour-vol",
            "order": "desc",
        }
        resp = await self._http.get("/markets", params=params)
        resp.raise_for_status()
        markets: list[Market] = []
        for item in resp.json():
            if item.get("outcomeType") != "BINARY":
                continue
            m = self._parse_market(item)
            if active_only and m.status != MarketStatus.OPEN:
                continue
            if m.volume_24h < min_volume:
                continue
            markets.append(m)
        return markets

    async def get_market(self, market_id: str) -> Market:
        resp = await self._http.get(f"/market/{market_id}")
        resp.raise_for_status()
        return self._parse_market(resp.json())

    async def get_price_history(
        self,
        market_id: str,
        *,
        points: int = 100,
    ) -> list[PricePoint]:
        resp = await self._http.get(f"/market/{market_id}")
        resp.raise_for_status()
        data = resp.json()

        # Manifold embeds bets as the price history source
        bets_resp = await self._http.get(
            "/bets",
            params={"contractId": market_id, "limit": points, "order": "desc"},
        )
        bets_resp.raise_for_status()
        history: list[PricePoint] = []
        for bet in reversed(bets_resp.json()):
            prob_after = bet.get("probAfter", data.get("probability", 0.5))
            history.append(
                PricePoint(
                    timestamp=datetime.fromtimestamp(bet["createdTime"] / 1000, tz=UTC),
                    probability=float(prob_after),
                    volume=abs(float(bet.get("amount", 0))),
                )
            )
        return history

    async def get_positions(self) -> list[Position]:
        # Manifold doesn't expose a direct positions endpoint in v0;
        # we approximate from the user's bets. Requires auth.
        resp = await self._http.get("/me")
        if resp.status_code != 200:
            return []
        # A full implementation would aggregate bets into positions.
        return []

    async def place_order(self, order: Order) -> str:
        outcome = "YES" if order.side == Side.YES else "NO"
        body = {
            "contractId": order.market_id,
            "outcome": outcome,
            "amount": order.amount,
        }
        if order.limit_price is not None:
            body["limitProb"] = order.limit_price
        resp = await self._http.post("/bet", json=body)
        resp.raise_for_status()
        return resp.json().get("betId", "")

    async def cancel_order(self, order_id: str) -> bool:
        resp = await self._http.post(f"/bet/cancel/{order_id}")
        return resp.status_code == 200

    @staticmethod
    def _parse_market(data: dict) -> Market:
        is_resolved = data.get("isResolved", False)
        close_ts = data.get("closeTime")
        if is_resolved:
            status = MarketStatus.RESOLVED
        elif close_ts and close_ts < int(datetime.now(UTC).timestamp() * 1000):
            status = MarketStatus.CLOSED
        else:
            status = MarketStatus.OPEN

        close_time = None
        if close_ts:
            close_time = datetime.fromtimestamp(close_ts / 1000, tz=UTC)

        created = None
        if data.get("createdTime"):
            created = datetime.fromtimestamp(data["createdTime"] / 1000, tz=UTC)

        return Market(
            id=data.get("id", ""),
            source="manifold",
            question=data.get("question", ""),
            url=data.get("url", ""),
            status=status,
            probability=float(data.get("probability", 0.5)),
            volume_24h=float(data.get("volume24Hours", 0)),
            liquidity=float(data.get("totalLiquidity", 0)),
            close_time=close_time,
            created_at=created,
        )

    async def close(self) -> None:
        await self._http.aclose()
