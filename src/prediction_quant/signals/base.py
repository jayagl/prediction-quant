"""Abstract base class for signal generators."""

from __future__ import annotations

import abc

from prediction_quant.models import Market, PricePoint, Signal


class SignalGenerator(abc.ABC):
    """Produces a trading signal for a given market."""

    name: str

    @abc.abstractmethod
    def generate(self, market: Market, history: list[PricePoint]) -> Signal | None:
        """Compute a signal from market data.

        Returns None if there is insufficient data to produce a signal.
        """
