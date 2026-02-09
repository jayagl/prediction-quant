"""Composite signal strategy.

Combines multiple signal generators into a single aggregated signal
and converts it into a trade decision when the aggregated score
exceeds a configurable threshold.
"""

from __future__ import annotations

from prediction_quant.models import (
    Market,
    OrderAction,
    Position,
    PricePoint,
    Side,
    Signal,
    TradeDecision,
)
from prediction_quant.signals.base import SignalGenerator
from prediction_quant.strategy.base import Strategy


class CompositeSignalStrategy(Strategy):
    """Aggregate signals with confidence-weighted averaging."""

    name = "composite_signal"

    def __init__(
        self,
        generators: list[SignalGenerator],
        entry_threshold: float = 0.30,
        exit_threshold: float = 0.10,
        base_amount: float = 10.0,
    ) -> None:
        self.generators = generators
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.base_amount = base_amount

    def compute_signals(self, market: Market, history: list[PricePoint]) -> list[Signal]:
        signals: list[Signal] = []
        for gen in self.generators:
            sig = gen.generate(market, history)
            if sig is not None:
                signals.append(sig)
        return signals

    def evaluate(
        self,
        market: Market,
        history: list[PricePoint],
        signals: list[Signal],
        position: Position | None,
    ) -> TradeDecision | None:
        if not signals:
            return None

        # Confidence-weighted average of signal values
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            return None
        agg_value = sum(s.value * s.confidence for s in signals) / total_weight
        avg_confidence = total_weight / len(signals)

        has_position = position is not None and position.shares > 0

        # --- Exit logic ---
        if has_position:
            assert position is not None
            # Exit if signal flipped against position
            if position.side == Side.YES and agg_value < -self.exit_threshold:
                return TradeDecision(
                    market_id=market.id,
                    source=market.source,
                    side=position.side,
                    action=OrderAction.SELL,
                    target_amount=position.shares,
                    signals=signals,
                    reason=f"Exit YES: aggregated signal={agg_value:.3f}",
                )
            if position.side == Side.NO and agg_value > self.exit_threshold:
                return TradeDecision(
                    market_id=market.id,
                    source=market.source,
                    side=position.side,
                    action=OrderAction.SELL,
                    target_amount=position.shares,
                    signals=signals,
                    reason=f"Exit NO: aggregated signal={agg_value:.3f}",
                )
            return None  # Hold

        # --- Entry logic ---
        if abs(agg_value) < self.entry_threshold:
            return None  # Not strong enough

        side = Side.YES if agg_value > 0 else Side.NO
        # Scale amount by conviction strength
        amount = self.base_amount * min(abs(agg_value) / self.entry_threshold, 3.0)
        # Use current market price as limit
        limit_price = market.probability if side == Side.YES else (1 - market.probability)

        return TradeDecision(
            market_id=market.id,
            source=market.source,
            side=side,
            action=OrderAction.BUY,
            target_amount=round(amount, 2),
            limit_price=limit_price,
            signals=signals,
            reason=f"Entry {side.value}: score={agg_value:.3f} conf={avg_confidence:.2f}",
        )
