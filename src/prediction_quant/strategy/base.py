"""Abstract base class for trading strategies."""

from __future__ import annotations

import abc

from prediction_quant.models import Market, Position, PricePoint, Signal, TradeDecision


class Strategy(abc.ABC):
    """Translates signals + market state into trade decisions."""

    name: str

    @abc.abstractmethod
    def evaluate(
        self,
        market: Market,
        history: list[PricePoint],
        signals: list[Signal],
        position: Position | None,
    ) -> TradeDecision | None:
        """Return a trade decision or None if no action should be taken."""
