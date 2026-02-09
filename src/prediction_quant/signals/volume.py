"""Volume-weighted directional signal.

Detects whether recent volume surges are accompanied by price moves,
producing a signal in the direction of the volume-weighted trend.
"""

from __future__ import annotations

import numpy as np

from prediction_quant.models import Market, PricePoint, Signal
from prediction_quant.signals.base import SignalGenerator


class VolumeSignal(SignalGenerator):
    """Signal based on volume-weighted price momentum."""

    name = "volume_momentum"

    def __init__(self, lookback: int = 10) -> None:
        self.lookback = lookback

    def generate(self, market: Market, history: list[PricePoint]) -> Signal | None:
        if len(history) < self.lookback + 1:
            return None

        recent = history[-self.lookback :]
        volumes = np.array([p.volume for p in recent])
        prices = np.array([p.probability for p in recent])
        deltas = np.diff(prices)

        total_vol = volumes[1:].sum()
        if total_vol == 0:
            # Fall back to unweighted momentum if no volume data
            raw = float(deltas.mean()) * 10
        else:
            # Volume-weighted average price change
            raw = float(np.dot(volumes[1:], deltas) / total_vol) * 10

        value = float(np.clip(raw, -1.0, 1.0))

        # Confidence scales with how much volume there is relative to 24h
        vol_ratio = min(total_vol / max(market.volume_24h, 1.0), 1.0)
        confidence = 0.3 + 0.5 * vol_ratio

        return Signal(
            name=self.name,
            market_id=market.id,
            value=value,
            confidence=confidence,
            metadata={
                "total_volume": float(total_vol),
                "avg_delta": float(deltas.mean()),
            },
        )
