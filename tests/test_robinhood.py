"""Tests for the Robinhood market client."""

import pytest
import respx
from httpx import Response

from prediction_quant.markets.robinhood import RobinhoodClient
from prediction_quant.models import MarketStatus


@pytest.fixture
def rh():
    return RobinhoodClient(access_token="test-token")


class TestRobinhoodParseMarket:
    def test_parse_open_market(self):
        data = {
            "id": "evt-123",
            "title": "Will X happen?",
            "state": "open",
            "last_trade_price": "0.65",
            "volume_24h": 500,
            "open_interest": 2000,
            "expiration_date": "2026-06-01T00:00:00Z",
        }
        m = RobinhoodClient._parse_market(data)
        assert m.id == "evt-123"
        assert m.source == "robinhood"
        assert m.status == MarketStatus.OPEN
        assert m.probability == pytest.approx(0.65)
        assert m.volume_24h == 500
        assert m.close_time is not None

    def test_parse_settled_market(self):
        data = {
            "id": "evt-456",
            "title": "Did Y happen?",
            "state": "settled",
            "last_trade_price": "0.95",
        }
        m = RobinhoodClient._parse_market(data)
        assert m.status == MarketStatus.RESOLVED

    def test_parse_cents_format(self):
        data = {
            "id": "evt-789",
            "title": "Test",
            "state": "trading",
            "last_trade_price": 72,
            "volume": 300,
        }
        m = RobinhoodClient._parse_market(data)
        assert m.status == MarketStatus.OPEN
        assert m.probability == pytest.approx(0.72)
        assert m.volume_24h == 300

    def test_parse_with_event_contract_id(self):
        data = {
            "event_contract_id": "ec-100",
            "display_name": "Some event",
            "status": "active",
            "probability": 0.5,
        }
        m = RobinhoodClient._parse_market(data)
        assert m.id == "ec-100"
        assert m.question == "Some event"

    def test_probability_clamped(self):
        data = {
            "id": "t",
            "title": "T",
            "state": "open",
            "last_trade_price": 150,
        }
        m = RobinhoodClient._parse_market(data)
        assert m.probability == 1.0


class TestRobinhoodAPI:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_markets_derivatives(self, rh):
        respx.get("https://derivatives.robinhood.com/api/v1/event-contracts/").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "id": "e1",
                            "title": "Event 1",
                            "state": "open",
                            "last_trade_price": "0.55",
                            "volume_24h": 100,
                        }
                    ]
                },
            )
        )
        markets = await rh.list_markets(limit=10)
        assert len(markets) == 1
        assert markets[0].id == "e1"
        await rh.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_markets_fallback(self, rh):
        respx.get("https://derivatives.robinhood.com/api/v1/event-contracts/").mock(
            return_value=Response(404)
        )
        respx.get("https://api.robinhood.com/options/events/").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "id": "e2",
                            "title": "Event 2",
                            "state": "open",
                            "last_trade_price": "0.40",
                            "volume_24h": 50,
                        }
                    ]
                },
            )
        )
        markets = await rh.list_markets()
        assert len(markets) == 1
        assert markets[0].id == "e2"
        await rh.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_markets_both_fail(self, rh):
        respx.get("https://derivatives.robinhood.com/api/v1/event-contracts/").mock(
            return_value=Response(403)
        )
        respx.get("https://api.robinhood.com/options/events/").mock(
            return_value=Response(403)
        )
        markets = await rh.list_markets()
        assert markets == []
        await rh.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_positions_empty(self, rh):
        respx.get("https://derivatives.robinhood.com/api/v1/event-contracts/positions/").mock(
            return_value=Response(401)
        )
        respx.get("https://api.robinhood.com/options/events/positions/").mock(
            return_value=Response(401)
        )
        positions = await rh.get_positions()
        assert positions == []
        await rh.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_price_history_empty(self, rh):
        respx.get(url__regex=r".*/event-contracts/m1/historicals/").mock(
            return_value=Response(404)
        )
        respx.get(url__regex=r".*/marketdata/events/m1/historicals/").mock(
            return_value=Response(404)
        )
        history = await rh.get_price_history("m1")
        assert history == []
        await rh.close()
