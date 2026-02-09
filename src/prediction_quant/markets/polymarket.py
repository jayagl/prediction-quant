"""Polymarket CLOB API client."""

from __future__ import annotations

import hashlib
import hmac
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

_BASE_URL = "https://clob.polymarket.com"
_GAMMA_URL = "https://gamma-api.polymarket.com"


class PolymarketClient(MarketClient):
    """Client for the Polymarket CLOB + Gamma APIs."""

    source = "polymarket"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        headers: dict[str, str] = {}
        if api_key:
            headers["POLY_API_KEY"] = api_key
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )
        self._gamma = httpx.AsyncClient(
            base_url=_GAMMA_URL,
            timeout=30.0,
        )

    # -- authentication helpers ------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = f"{timestamp}{method}{path}{body}"
        mac = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        )
        return mac.hexdigest()

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts = str(int(time.time()))
        return {
            "POLY_API_KEY": self._api_key,
            "POLY_SIGNATURE": self._sign(ts, method, path, body),
            "POLY_TIMESTAMP": ts,
            "POLY_PASSPHRASE": self._api_passphrase,
        }

    # -- public data -----------------------------------------------------------

    async def list_markets(
        self,
        *,
        limit: int = 50,
        active_only: bool = True,
        min_volume: float = 0.0,
    ) -> list[Market]:
        params: dict[str, str | int | bool] = {
            "limit": limit,
            "active": active_only,
            "order": "volume24hr",
            "ascending": False,
        }
        resp = await self._gamma.get("/markets", params=params)
        resp.raise_for_status()
        markets: list[Market] = []
        for item in resp.json():
            vol = float(item.get("volume24hr", 0))
            if vol < min_volume:
                continue
            markets.append(self._parse_market(item))
        return markets

    async def get_market(self, market_id: str) -> Market:
        resp = await self._gamma.get(f"/markets/{market_id}")
        resp.raise_for_status()
        return self._parse_market(resp.json())

    async def get_price_history(
        self,
        market_id: str,
        *,
        points: int = 100,
    ) -> list[PricePoint]:
        resp = await self._gamma.get(
            f"/markets/{market_id}/prices",
            params={"limit": points, "interval": "1h"},
        )
        resp.raise_for_status()
        history: list[PricePoint] = []
        for pt in resp.json().get("history", []):
            history.append(
                PricePoint(
                    timestamp=datetime.fromtimestamp(pt["t"], tz=UTC),
                    probability=float(pt["p"]),
                )
            )
        return history

    # -- authenticated ---------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        if not self._api_key:
            return []
        path = "/positions"
        headers = self._auth_headers("GET", path)
        resp = await self._http.get(path, headers=headers)
        resp.raise_for_status()
        positions: list[Position] = []
        for p in resp.json():
            positions.append(
                Position(
                    market_id=p["asset_id"],
                    source=self.source,
                    side=Side.YES if p.get("side", "YES") == "YES" else Side.NO,
                    shares=float(p.get("size", 0)),
                    avg_price=float(p.get("avg_price", 0)),
                    current_price=float(p.get("cur_price", 0)),
                )
            )
        return positions

    async def place_order(self, order: Order) -> str:
        path = "/order"
        body_dict = {
            "tokenID": order.market_id,
            "side": "BUY" if order.action == OrderAction.BUY else "SELL",
            "type": "GTC",
            "size": str(order.amount),
        }
        if order.limit_price is not None:
            body_dict["price"] = str(order.limit_price)
        import json

        body = json.dumps(body_dict)
        headers = self._auth_headers("POST", path, body)
        headers["Content-Type"] = "application/json"
        resp = await self._http.post(path, content=body, headers=headers)
        resp.raise_for_status()
        return resp.json().get("orderID", "")

    async def cancel_order(self, order_id: str) -> bool:
        path = f"/order/{order_id}"
        headers = self._auth_headers("DELETE", path)
        resp = await self._http.delete(path, headers=headers)
        return resp.status_code == 200

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _parse_market(data: dict) -> Market:
        status_map = {
            "active": MarketStatus.OPEN,
            "closed": MarketStatus.CLOSED,
            "resolved": MarketStatus.RESOLVED,
        }
        raw_status = data.get("active", True)
        if isinstance(raw_status, bool):
            status = MarketStatus.OPEN if raw_status else MarketStatus.CLOSED
        else:
            status = status_map.get(str(raw_status).lower(), MarketStatus.OPEN)

        close_time = None
        if data.get("endDate"):
            try:
                close_time = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Market(
            id=str(data.get("id", data.get("condition_id", ""))),
            source="polymarket",
            question=data.get("question", ""),
            url=f"https://polymarket.com/event/{data.get('slug', '')}",
            status=status,
            probability=float(data.get("outcomePrices", [0.5])[0] if isinstance(data.get("outcomePrices"), list) else data.get("probability", 0.5)),
            volume_24h=float(data.get("volume24hr", 0)),
            liquidity=float(data.get("liquidity", 0)),
            close_time=close_time,
        )

    async def close(self) -> None:
        await self._http.aclose()
        await self._gamma.aclose()
