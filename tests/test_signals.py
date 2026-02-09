"""Tests for signal generators."""

from datetime import UTC, datetime, timedelta

from prediction_quant.models import Market, MarketStatus, PricePoint
from prediction_quant.signals.ema import EMACrossoverSignal
from prediction_quant.signals.fair_value import FairValueDivergenceSignal
from prediction_quant.signals.volume import VolumeSignal


def _make_market(probability: float = 0.5, volume_24h: float = 1000.0) -> Market:
    return Market(
        id="test-market",
        source="test",
        question="Will it rain tomorrow?",
        probability=probability,
        volume_24h=volume_24h,
        status=MarketStatus.OPEN,
    )


def _make_history(
    probabilities: list[float],
    volumes: list[float] | None = None,
) -> list[PricePoint]:
    now = datetime.now(UTC)
    if volumes is None:
        volumes = [10.0] * len(probabilities)
    return [
        PricePoint(
            timestamp=now - timedelta(hours=len(probabilities) - i),
            probability=p,
            volume=v,
        )
        for i, (p, v) in enumerate(zip(probabilities, volumes))
    ]


class TestEMACrossoverSignal:
    def test_insufficient_data_returns_none(self):
        sig = EMACrossoverSignal(fast_span=5, slow_span=20)
        market = _make_market()
        history = _make_history([0.5] * 10)  # Less than slow_span+1
        assert sig.generate(market, history) is None

    def test_uptrend_produces_positive_signal(self):
        sig = EMACrossoverSignal(fast_span=3, slow_span=10)
        market = _make_market()
        # Steadily rising from 0.3 to 0.7
        prices = [0.3 + i * 0.02 for i in range(25)]
        history = _make_history(prices)
        result = sig.generate(market, history)
        assert result is not None
        assert result.value > 0
        assert result.name == "ema_crossover"

    def test_downtrend_produces_negative_signal(self):
        sig = EMACrossoverSignal(fast_span=3, slow_span=10)
        market = _make_market()
        # Steadily falling from 0.7 to 0.3
        prices = [0.7 - i * 0.02 for i in range(25)]
        history = _make_history(prices)
        result = sig.generate(market, history)
        assert result is not None
        assert result.value < 0

    def test_flat_produces_near_zero_signal(self):
        sig = EMACrossoverSignal(fast_span=5, slow_span=20)
        market = _make_market()
        history = _make_history([0.50] * 25)
        result = sig.generate(market, history)
        assert result is not None
        assert abs(result.value) < 0.05


class TestVolumeSignal:
    def test_insufficient_data_returns_none(self):
        sig = VolumeSignal(lookback=10)
        market = _make_market()
        history = _make_history([0.5] * 5)
        assert sig.generate(market, history) is None

    def test_volume_weighted_uptrend(self):
        sig = VolumeSignal(lookback=10)
        market = _make_market()
        prices = [0.40 + i * 0.01 for i in range(15)]
        volumes = [100.0] * 15
        history = _make_history(prices, volumes)
        result = sig.generate(market, history)
        assert result is not None
        assert result.value > 0

    def test_zero_volume_fallback(self):
        sig = VolumeSignal(lookback=10)
        market = _make_market()
        prices = [0.40 + i * 0.01 for i in range(15)]
        volumes = [0.0] * 15
        history = _make_history(prices, volumes)
        result = sig.generate(market, history)
        assert result is not None
        # Should still produce a signal based on unweighted momentum


class TestFairValueDivergenceSignal:
    def test_insufficient_data(self):
        sig = FairValueDivergenceSignal(lookback=30)
        market = _make_market()
        history = _make_history([0.5] * 10)
        assert sig.generate(market, history) is None

    def test_underpriced_market(self):
        sig = FairValueDivergenceSignal(lookback=20)
        # History suggests ~0.6, but market is at 0.45
        market = _make_market(probability=0.45)
        prices = [0.58 + i * 0.002 for i in range(20)]
        history = _make_history(prices)
        result = sig.generate(market, history)
        assert result is not None
        assert result.value > 0  # Fair value > market -> buy YES

    def test_overpriced_market(self):
        sig = FairValueDivergenceSignal(lookback=20)
        # History suggests ~0.4, but market is at 0.55
        market = _make_market(probability=0.55)
        prices = [0.42 - i * 0.002 for i in range(20)]
        history = _make_history(prices)
        result = sig.generate(market, history)
        assert result is not None
        assert result.value < 0  # Fair value < market -> buy NO
