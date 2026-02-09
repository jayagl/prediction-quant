"""Tests for the composite strategy."""

from datetime import UTC, datetime, timedelta

from prediction_quant.models import (
    Market,
    MarketStatus,
    OrderAction,
    Position,
    PricePoint,
    Side,
    Signal,
)
from prediction_quant.signals.ema import EMACrossoverSignal
from prediction_quant.signals.volume import VolumeSignal
from prediction_quant.strategy.composite import CompositeSignalStrategy


def _make_market(probability: float = 0.5) -> Market:
    return Market(
        id="m1",
        source="test",
        question="Test?",
        probability=probability,
        status=MarketStatus.OPEN,
    )


def _make_history(n: int = 30, base: float = 0.5) -> list[PricePoint]:
    now = datetime.now(UTC)
    return [
        PricePoint(
            timestamp=now - timedelta(hours=n - i),
            probability=base,
            volume=10.0,
        )
        for i in range(n)
    ]


class TestCompositeSignalStrategy:
    def test_no_signals_returns_none(self):
        strat = CompositeSignalStrategy(generators=[])
        market = _make_market()
        assert strat.evaluate(market, [], [], None) is None

    def test_weak_signal_no_trade(self):
        strat = CompositeSignalStrategy(
            generators=[],
            entry_threshold=0.50,
        )
        market = _make_market()
        signals = [
            Signal(name="test", market_id="m1", value=0.2, confidence=0.5),
        ]
        assert strat.evaluate(market, [], signals, None) is None

    def test_strong_positive_signal_buys_yes(self):
        strat = CompositeSignalStrategy(
            generators=[],
            entry_threshold=0.30,
            base_amount=10.0,
        )
        market = _make_market(probability=0.45)
        signals = [
            Signal(name="s1", market_id="m1", value=0.8, confidence=0.9),
            Signal(name="s2", market_id="m1", value=0.6, confidence=0.7),
        ]
        decision = strat.evaluate(market, [], signals, None)
        assert decision is not None
        assert decision.side == Side.YES
        assert decision.action == OrderAction.BUY
        assert decision.target_amount > 0

    def test_strong_negative_signal_buys_no(self):
        strat = CompositeSignalStrategy(
            generators=[],
            entry_threshold=0.30,
            base_amount=10.0,
        )
        market = _make_market(probability=0.55)
        signals = [
            Signal(name="s1", market_id="m1", value=-0.7, confidence=0.8),
        ]
        decision = strat.evaluate(market, [], signals, None)
        assert decision is not None
        assert decision.side == Side.NO
        assert decision.action == OrderAction.BUY

    def test_exit_on_signal_reversal(self):
        strat = CompositeSignalStrategy(
            generators=[],
            exit_threshold=0.10,
        )
        market = _make_market()
        position = Position(
            market_id="m1",
            source="test",
            side=Side.YES,
            shares=5.0,
            avg_price=0.40,
            current_price=0.50,
        )
        # Strong negative signal while holding YES
        signals = [
            Signal(name="s1", market_id="m1", value=-0.5, confidence=0.8),
        ]
        decision = strat.evaluate(market, [], signals, position)
        assert decision is not None
        assert decision.action == OrderAction.SELL
        assert decision.target_amount == 5.0

    def test_hold_when_signal_aligned(self):
        strat = CompositeSignalStrategy(
            generators=[],
            exit_threshold=0.10,
        )
        market = _make_market()
        position = Position(
            market_id="m1",
            source="test",
            side=Side.YES,
            shares=5.0,
            avg_price=0.40,
            current_price=0.50,
        )
        # Positive signal aligns with YES position
        signals = [
            Signal(name="s1", market_id="m1", value=0.4, confidence=0.7),
        ]
        decision = strat.evaluate(market, [], signals, position)
        assert decision is None  # Hold

    def test_compute_signals_from_generators(self):
        strat = CompositeSignalStrategy(
            generators=[EMACrossoverSignal(fast_span=3, slow_span=10), VolumeSignal(lookback=5)],
        )
        market = _make_market()
        # Rising trend
        now = datetime.now(UTC)
        history = [
            PricePoint(
                timestamp=now - timedelta(hours=30 - i),
                probability=0.40 + i * 0.01,
                volume=50.0,
            )
            for i in range(30)
        ]
        signals = strat.compute_signals(market, history)
        assert len(signals) == 2
        assert all(s.market_id == "m1" for s in signals)
