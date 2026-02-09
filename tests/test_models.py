"""Tests for domain models."""

import pytest

from prediction_quant.models import Position, Side


class TestPosition:
    def test_market_value(self):
        pos = Position(
            market_id="m1",
            source="test",
            side=Side.YES,
            shares=10.0,
            avg_price=0.40,
            current_price=0.60,
        )
        assert pos.market_value == 6.0

    def test_unrealised_pnl(self):
        pos = Position(
            market_id="m1",
            source="test",
            side=Side.YES,
            shares=10.0,
            avg_price=0.40,
            current_price=0.60,
        )
        assert pos.unrealised_pnl == pytest.approx(2.0)

    def test_zero_shares(self):
        pos = Position(
            market_id="m1",
            source="test",
            side=Side.NO,
            shares=0.0,
            avg_price=0.50,
            current_price=0.50,
        )
        assert pos.market_value == 0.0
        assert pos.unrealised_pnl == 0.0
