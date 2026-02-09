"""Kalshi exchange API client.

Uses the official Kalshi REST API (v2) with RSA-PSS request signing.
Docs: https://docs.kalshi.com
"""

from __future__ import annotations

import base64
import json
import logging
import time
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

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"


def _load_private_key(path: str):
    """Load an RSA private key from a PEM file.

    Returns the key object, or None if the cryptography library
    is not installed or the file is missing.
    """
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError:
        logger.warning(
            "cryptography package not installed — Kalshi auth will not work. "
            "Install with: pip install cryptography"
        )
        return None

    try:
        with open(path, "rb") as f:
            return load_pem_private_key(f.read(), password=None)
    except FileNotFoundError:
        logger.warning("Kalshi private key file not found: %s", path)
        return None


class KalshiClient(MarketClient):
    """Client for the Kalshi exchange REST API (v2)."""

    source = "kalshi"

    def __init__(
        self,
        api_key: str = "",
        private_key_path: str = "",
        demo: bool = False,
    ) -> None:
        self._api_key = api_key
        self._private_key = _load_private_key(private_key_path) if private_key_path else None
        base_url = _DEMO_URL if demo else _BASE_URL
        self._base_url = base_url
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
        )

    # -- authentication --------------------------------------------------------

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """Produce an RSA-PSS signature for the request."""
        if self._private_key is None:
            return ""

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        message = f"{timestamp_ms}{method}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    # -- public data -----------------------------------------------------------

    async def list_markets(
        self,
        *,
        limit: int = 50,
        active_only: bool = True,
        min_volume: float = 0.0,
    ) -> list[Market]:
        params: dict[str, str | int] = {"limit": min(limit, 200)}
        if active_only:
            params["status"] = "open"
        resp = await self._http.get("/markets", params=params)
        resp.raise_for_status()
        markets: list[Market] = []
        for item in resp.json().get("markets", []):
            m = self._parse_market(item)
            if m.volume_24h >= min_volume:
                markets.append(m)
        return markets

    async def get_market(self, market_id: str) -> Market:
        resp = await self._http.get(f"/markets/{market_id}")
        resp.raise_for_status()
        return self._parse_market(resp.json().get("market", resp.json()))

    async def get_price_history(
        self,
        market_id: str,
        *,
        points: int = 100,
    ) -> list[PricePoint]:
        # Kalshi candlestick endpoint requires series_ticker.
        # Extract it from the market ticker (e.g. "FED-26MAR-T4.50" -> series "FED").
        # For simplicity, try the market ticker directly first.
        parts = market_id.split("-")
        series_ticker = parts[0] if parts else market_id

        resp = await self._http.get(
            f"/series/{series_ticker}/markets/{market_id}/candlesticks",
            params={"period_interval": "1h"},
        )
        if resp.status_code != 200:
            # Fallback: try the batch endpoint
            resp = await self._http.get(
                "/markets/candlesticks",
                params={
                    "tickers": market_id,
                    "period_interval": "1h",
                },
            )
            if resp.status_code != 200:
                return []

        history: list[PricePoint] = []
        candles = resp.json().get("candlesticks", [])
        for c in candles[-points:]:
            ts = c.get("end_period_ts") or c.get("start_period_ts", 0)
            # Prices are in cents (1-99); convert to 0.0-1.0
            close_price = float(c.get("yes_price", {}).get("close", 50)) / 100.0
            volume = float(c.get("volume", 0))
            history.append(
                PricePoint(
                    timestamp=datetime.fromtimestamp(ts, tz=UTC) if ts else datetime.now(UTC),
                    probability=close_price,
                    volume=volume,
                )
            )
        return history

    # -- authenticated ---------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        if not self._api_key:
            return []
        path = "/portfolio/positions"
        headers = self._auth_headers("GET", path)
        resp = await self._http.get(path, headers=headers)
        if resp.status_code != 200:
            return []
        positions: list[Position] = []
        for mp in resp.json().get("market_positions", []):
            position_count = float(mp.get("position", 0))
            if position_count == 0:
                continue
            side = Side.YES if position_count > 0 else Side.NO
            exposure_dollars = mp.get("market_exposure_dollars", "0")
            shares = abs(position_count)
            avg_price = float(exposure_dollars) / shares if shares else 0.0
            positions.append(
                Position(
                    market_id=mp.get("ticker", ""),
                    source=self.source,
                    side=side,
                    shares=shares,
                    avg_price=avg_price,
                    current_price=avg_price,  # Updated on next market fetch
                )
            )
        return positions

    async def place_order(self, order: Order) -> str:
        path = "/portfolio/orders"
        # Convert amount (dollars) to count (contracts) at the limit price
        price_cents = int((order.limit_price or 0.50) * 100)
        count = max(1, int(order.amount / (price_cents / 100.0)))

        body = {
            "ticker": order.market_id,
            "side": "yes" if order.side == Side.YES else "no",
            "action": "buy" if order.action == OrderAction.BUY else "sell",
            "type": "limit" if order.limit_price is not None else "market",
            "count": count,
            "time_in_force": "good_till_canceled",
        }
        if order.limit_price is not None:
            body["yes_price"] = price_cents

        headers = self._auth_headers("POST", path)
        resp = await self._http.post(path, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json().get("order", {}).get("order_id", "")

    async def cancel_order(self, order_id: str) -> bool:
        path = f"/portfolio/orders/{order_id}"
        headers = self._auth_headers("DELETE", path)
        resp = await self._http.delete(path, headers=headers)
        return resp.status_code == 200

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _parse_market(data: dict) -> Market:
        status_map = {
            "open": MarketStatus.OPEN,
            "active": MarketStatus.OPEN,
            "closed": MarketStatus.CLOSED,
            "paused": MarketStatus.CLOSED,
            "settled": MarketStatus.RESOLVED,
            "finalized": MarketStatus.RESOLVED,
        }
        raw_status = data.get("status", "open")
        status = status_map.get(raw_status, MarketStatus.OPEN)

        # Probability from last_price_dollars or yes_bid
        prob = 0.5
        if data.get("last_price_dollars"):
            prob = float(data["last_price_dollars"])
        elif data.get("yes_bid_dollars"):
            yes_bid = float(data["yes_bid_dollars"])
            yes_ask = float(data.get("yes_ask_dollars", yes_bid))
            prob = (yes_bid + yes_ask) / 2.0
        elif data.get("last_price"):
            prob = float(data["last_price"]) / 100.0

        close_time = None
        if data.get("close_time"):
            try:
                close_time = datetime.fromisoformat(
                    data["close_time"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        volume_24h = 0.0
        if data.get("volume_24h_fp"):
            volume_24h = float(data["volume_24h_fp"])
        elif data.get("volume_24h"):
            volume_24h = float(data["volume_24h"])

        liquidity = float(data.get("open_interest_fp", data.get("open_interest", 0)))

        return Market(
            id=data.get("ticker", ""),
            source="kalshi",
            question=data.get("title", data.get("subtitle", "")),
            url=f"https://kalshi.com/markets/{data.get('ticker', '')}",
            status=status,
            probability=max(0.0, min(1.0, prob)),
            volume_24h=volume_24h,
            liquidity=liquidity,
            close_time=close_time,
        )

    async def close(self) -> None:
        await self._http.aclose()
