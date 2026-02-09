"""Tests for the Kalshi market client."""

import pytest
import respx
from httpx import Response

from prediction_quant.markets.kalshi import KalshiClient
from prediction_quant.models import MarketStatus, Order, OrderAction, Side


@pytest.fixture
def kalshi():
    return KalshiClient(api_key="test-key", private_key_path="")


class TestKalshiParseMarket:
    def test_parse_open_market(self):
        data = {
            "ticker": "FED-26MAR-T4.50",
            "title": "Fed holds rates at 4.50%?",
            "status": "open",
            "last_price_dollars": "0.5500",
            "volume_24h_fp": "1234.00",
            "open_interest_fp": "5000.00",
            "close_time": "2026-03-19T18:00:00Z",
        }
        m = KalshiClient._parse_market(data)
        assert m.id == "FED-26MAR-T4.50"
        assert m.source == "kalshi"
        assert m.status == MarketStatus.OPEN
        assert m.probability == pytest.approx(0.55)
        assert m.volume_24h == pytest.approx(1234.0)
        assert m.liquidity == pytest.approx(5000.0)
        assert m.close_time is not None

    def test_parse_settled_market(self):
        data = {
            "ticker": "PRES-2024",
            "title": "Who wins 2024 election?",
            "status": "settled",
            "last_price": 85,
            "volume_24h": 500,
        }
        m = KalshiClient._parse_market(data)
        assert m.status == MarketStatus.RESOLVED
        assert m.probability == pytest.approx(0.85)

    def test_parse_bid_ask_midpoint(self):
        data = {
            "ticker": "TEST-MKT",
            "title": "Test",
            "status": "open",
            "yes_bid_dollars": "0.40",
            "yes_ask_dollars": "0.50",
        }
        m = KalshiClient._parse_market(data)
        assert m.probability == pytest.approx(0.45)

    def test_probability_clamped(self):
        data = {
            "ticker": "T",
            "title": "T",
            "status": "open",
            "last_price_dollars": "1.50",
        }
        m = KalshiClient._parse_market(data)
        assert m.probability == 1.0


class TestKalshiAPI:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_markets(self, kalshi):
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "M1",
                            "title": "Market 1",
                            "status": "open",
                            "last_price_dollars": "0.60",
                            "volume_24h_fp": "100.00",
                        },
                        {
                            "ticker": "M2",
                            "title": "Market 2",
                            "status": "open",
                            "last_price_dollars": "0.30",
                            "volume_24h_fp": "50.00",
                        },
                    ]
                },
            )
        )
        markets = await kalshi.list_markets(limit=10)
        assert len(markets) == 2
        assert markets[0].id == "M1"
        assert markets[1].probability == pytest.approx(0.30)
        await kalshi.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_markets_min_volume(self, kalshi):
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets").mock(
            return_value=Response(
                200,
                json={
                    "markets": [
                        {
                            "ticker": "M1",
                            "title": "Market 1",
                            "status": "open",
                            "last_price_dollars": "0.50",
                            "volume_24h_fp": "10.00",
                        },
                    ]
                },
            )
        )
        markets = await kalshi.list_markets(min_volume=100)
        assert len(markets) == 0
        await kalshi.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_market(self, kalshi):
        respx.get("https://api.elections.kalshi.com/trade-api/v2/markets/M1").mock(
            return_value=Response(
                200,
                json={
                    "market": {
                        "ticker": "M1",
                        "title": "Test Market",
                        "status": "open",
                        "last_price_dollars": "0.70",
                    }
                },
            )
        )
        m = await kalshi.get_market("M1")
        assert m.id == "M1"
        assert m.probability == pytest.approx(0.70)
        await kalshi.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_positions_no_key(self):
        client = KalshiClient(api_key="")
        positions = await client.get_positions()
        assert positions == []
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_price_history_fallback(self, kalshi):
        # First candlestick call fails, batch also fails
        respx.get(url__regex=r".*/series/.*/markets/.*/candlesticks").mock(
            return_value=Response(404)
        )
        respx.get(url__regex=r".*/markets/candlesticks").mock(
            return_value=Response(404)
        )
        history = await kalshi.get_price_history("M1")
        assert history == []
        await kalshi.close()
