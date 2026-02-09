"""Fair-value divergence signal.

Estimates a fair value for the probability using a simple Bayesian
update over recent price history and compares it with the current
market price. When the market price diverges from the estimated
fair value, there may be a trading opportunity.
"""

from __future__ import annotations

import numpy as np

from prediction_quant.models import Market, PricePoint, Signal
from prediction_quant.signals.base import SignalGenerator


class FairValueDivergenceSignal(SignalGenerator):
    """Signal based on divergence between market price and estimated fair value."""

    name = "fair_value_divergence"

    def __init__(self, lookback: int = 30, decay: float = 0.95) -> None:
        self.lookback = lookback
        self.decay = decay

    def generate(self, market: Market, history: list[PricePoint]) -> Signal | None:
        if len(history) < self.lookback:
            return None

        recent = history[-self.lookback :]
        prices = np.array([p.probability for p in recent])

        # Exponentially-weighted mean as fair value estimate
        weights = np.array([self.decay ** (self.lookback - 1 - i) for i in range(self.lookback)])
        weights /= weights.sum()
        fair_value = float(np.dot(weights, prices))

        divergence = fair_value - market.probability

        # Normalise: a 0.10 divergence is a strong signal
        value = float(np.clip(divergence / 0.10, -1.0, 1.0))

        # Confidence based on consistency of the fair value estimate
        std = float(np.sqrt(np.dot(weights, (prices - fair_value) ** 2)))
        confidence = float(np.clip(1.0 - std * 5, 0.2, 0.9))

        return Signal(
            name=self.name,
            market_id=market.id,
            value=value,
            confidence=confidence,
            metadata={
                "fair_value": fair_value,
                "market_price": market.probability,
                "divergence": divergence,
                "weighted_std": std,
            },
        )
