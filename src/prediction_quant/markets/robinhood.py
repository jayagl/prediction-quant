"""Robinhood event contracts client (unofficial/experimental).

Robinhood does not publish an official API for event contracts (prediction
markets).  Their event contracts are routed through KalshiEX LLC, so the
Kalshi client gives access to the same underlying markets with an official,
documented API.

This client uses the known Robinhood OAuth patterns adapted for event
contracts.  It is best-effort and may break without notice if Robinhood
changes their internal API surface.
"""

from __future__ import annotations

import logging
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

_BASE_URL = "https://api.robinhood.com"
_DERIVATIVES_URL = "https://derivatives.robinhood.com/api/v1"


class RobinhoodClient(MarketClient):
    """Experimental client for Robinhood event contracts.

    Authentication requires a bearer token obtained through the standard
    Robinhood OAuth flow (username/password + MFA).  This client does NOT
    handle the login flow itself — you must provide a valid access token.

    NOTE: Robinhood's event contracts are powered by KalshiEX.  For a
    reliable, documented integration consider using KalshiClient instead.
    """

    source = "robinhood"

    def __init__(self, access_token: str = "") -> None:
        headers: dict[str, str] = {"Accept": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self._http = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )
        self._deriv = httpx.AsyncClient(
            base_url=_DERIVATIVES_URL,
            headers=headers,
            timeout=30.0,
        )

    # -- public data -----------------------------------------------------------

    async def list_markets(
        self,
        *,
        limit: int = 50,
        active_only: bool = True,
        min_volume: float = 0.0,
    ) -> list[Market]:
        # Robinhood groups event contracts under "events"
        params: dict[str, str | int] = {"limit": limit}
        if active_only:
            params["state"] = "open"

        # Try the derivatives/event-contracts endpoint
        resp = await self._deriv.get("/event-contracts/", params=params)
        if resp.status_code != 200:
            # Fallback: try the main API surface
            resp = await self._http.get("/options/events/", params=params)
            if resp.status_code != 200:
                logger.warning(
                    "Robinhood event contracts endpoint returned %d", resp.status_code
                )
                return []

        results = resp.json().get("results", resp.json().get("items", []))
        markets: list[Market] = []
        for item in results:
            m = self._parse_market(item)
            if m.volume_24h >= min_volume:
                markets.append(m)
        return markets

    async def get_market(self, market_id: str) -> Market:
        resp = await self._deriv.get(f"/event-contracts/{market_id}/")
        if resp.status_code != 200:
            resp = await self._http.get(f"/options/events/{market_id}/")
            resp.raise_for_status()
        return self._parse_market(resp.json())

    async def get_price_history(
        self,
        market_id: str,
        *,
        points: int = 100,
    ) -> list[PricePoint]:
        # Try to get historical prices from the historicals endpoint
        resp = await self._deriv.get(
            f"/event-contracts/{market_id}/historicals/",
            params={"interval": "hour", "span": "week"},
        )
        if resp.status_code != 200:
            resp = await self._http.get(
                f"/marketdata/events/{market_id}/historicals/",
                params={"interval": "hour", "span": "week"},
            )
            if resp.status_code != 200:
                return []

        data = resp.json()
        data_points = data.get("data_points", data.get("historicals", []))
        history: list[PricePoint] = []
        for pt in data_points[-points:]:
            ts_str = pt.get("begins_at") or pt.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                ts = datetime.now(UTC)

            prob = float(pt.get("close_price", pt.get("last_trade_price", 0.5)))
            # Robinhood uses $0.01 - $0.99 range like Kalshi
            if prob > 1.0:
                prob = prob / 100.0
            volume = float(pt.get("volume", 0))
            history.append(PricePoint(timestamp=ts, probability=prob, volume=volume))
        return history

    # -- authenticated ---------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        resp = await self._deriv.get("/event-contracts/positions/")
        if resp.status_code != 200:
            resp = await self._http.get("/options/events/positions/")
            if resp.status_code != 200:
                return []

        results = resp.json().get("results", [])
        positions: list[Position] = []
        for p in results:
            quantity = float(p.get("quantity", 0))
            if quantity == 0:
                continue
            side_str = p.get("side", p.get("type", "yes")).upper()
            side = Side.YES if side_str == "YES" else Side.NO
            avg_price = float(p.get("average_price", 0))
            if avg_price > 1.0:
                avg_price /= 100.0
            positions.append(
                Position(
                    market_id=p.get("event_contract_id", p.get("id", "")),
                    source=self.source,
                    side=side,
                    shares=quantity,
                    avg_price=avg_price,
                    current_price=avg_price,
                )
            )
        return positions

    async def place_order(self, order: Order) -> str:
        body = {
            "event_contract_id": order.market_id,
            "side": "yes" if order.side == Side.YES else "no",
            "action": "buy" if order.action == OrderAction.BUY else "sell",
            "quantity": max(1, int(order.amount)),
            "type": "limit" if order.limit_price is not None else "market",
        }
        if order.limit_price is not None:
            # Convert probability to cents
            body["price"] = str(int(order.limit_price * 100))

        resp = await self._deriv.post("/event-contracts/orders/", json=body)
        if resp.status_code not in (200, 201):
            resp = await self._http.post("/options/events/orders/", json=body)
            resp.raise_for_status()

        return resp.json().get("id", resp.json().get("order_id", ""))

    async def cancel_order(self, order_id: str) -> bool:
        resp = await self._deriv.post(f"/event-contracts/orders/{order_id}/cancel/")
        if resp.status_code != 200:
            resp = await self._http.post(
                f"/options/events/orders/{order_id}/cancel/"
            )
        return resp.status_code == 200

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _parse_market(data: dict) -> Market:
        raw_state = data.get("state", data.get("status", "open")).lower()
        status_map = {
            "open": MarketStatus.OPEN,
            "active": MarketStatus.OPEN,
            "trading": MarketStatus.OPEN,
            "closed": MarketStatus.CLOSED,
            "settled": MarketStatus.RESOLVED,
            "resolved": MarketStatus.RESOLVED,
            "expired": MarketStatus.RESOLVED,
        }
        status = status_map.get(raw_state, MarketStatus.OPEN)

        prob = float(data.get("last_trade_price", data.get("probability", 0.5)))
        if prob > 1.0:
            prob /= 100.0

        close_time = None
        for field in ("expiration_date", "close_time", "expires_at"):
            if data.get(field):
                try:
                    close_time = datetime.fromisoformat(
                        data[field].replace("Z", "+00:00")
                    )
                    break
                except (ValueError, TypeError):
                    pass

        volume_24h = float(data.get("volume_24h", data.get("volume", 0)))

        return Market(
            id=data.get("id", data.get("event_contract_id", "")),
            source="robinhood",
            question=data.get("title", data.get("question", data.get("display_name", ""))),
            url=data.get("url", ""),
            status=status,
            probability=max(0.0, min(1.0, prob)),
            volume_24h=volume_24h,
            liquidity=float(data.get("open_interest", 0)),
            close_time=close_time,
        )

    async def close(self) -> None:
        await self._http.aclose()
        await self._deriv.aclose()
