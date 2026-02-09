"""Exponential Moving Average crossover signal.

Compares a fast EMA against a slow EMA of probability prices.
When the fast EMA crosses above the slow EMA the signal is bullish (YES),
and vice-versa. The magnitude is proportional to the spread.
"""

from __future__ import annotations

import numpy as np

from prediction_quant.models import Market, PricePoint, Signal
from prediction_quant.signals.base import SignalGenerator


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Compute EMA using the standard smoothing formula."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


class EMACrossoverSignal(SignalGenerator):
    """Signal based on fast/slow EMA crossover of probability."""

    name = "ema_crossover"

    def __init__(self, fast_span: int = 5, slow_span: int = 20) -> None:
        self.fast_span = fast_span
        self.slow_span = slow_span

    def generate(self, market: Market, history: list[PricePoint]) -> Signal | None:
        if len(history) < self.slow_span + 1:
            return None

        prices = np.array([p.probability for p in history])
        fast = _ema(prices, self.fast_span)
        slow = _ema(prices, self.slow_span)

        spread = fast[-1] - slow[-1]
        prev_spread = fast[-2] - slow[-2]

        # Normalise spread to [-1, 1] — cap at +-0.20 spread
        value = float(np.clip(spread / 0.20, -1.0, 1.0))

        # Higher confidence when crossing zero (regime change)
        crossing = (spread > 0) != (prev_spread > 0)
        confidence = 0.8 if crossing else 0.5

        return Signal(
            name=self.name,
            market_id=market.id,
            value=value,
            confidence=confidence,
            metadata={
                "fast_ema": float(fast[-1]),
                "slow_ema": float(slow[-1]),
                "spread": float(spread),
            },
        )
