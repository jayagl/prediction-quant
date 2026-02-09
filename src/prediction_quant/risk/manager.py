"""Risk management: position sizing, exposure limits, drawdown protection."""

from __future__ import annotations

import logging

from prediction_quant.models import Order, OrderAction, Position, TradeDecision

logger = logging.getLogger(__name__)


class RiskManager:
    """Gate-keeps trade decisions through risk rules before execution."""

    def __init__(
        self,
        max_position_size: float = 100.0,
        max_total_exposure: float = 1000.0,
        max_drawdown_pct: float = 0.10,
    ) -> None:
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_pct = max_drawdown_pct
        self._peak_value: float = 0.0
        self._current_value: float = 0.0

    def update_portfolio_value(self, value: float) -> None:
        """Track portfolio value for drawdown calculation."""
        self._current_value = value
        if value > self._peak_value:
            self._peak_value = value

    @property
    def current_drawdown(self) -> float:
        if self._peak_value == 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value

    def check(
        self,
        decision: TradeDecision,
        positions: list[Position],
    ) -> Order | None:
        """Validate a trade decision against risk limits.

        Returns a sized Order if approved, or None if the trade is rejected.
        """
        # Rule 1: drawdown halt
        if self.current_drawdown >= self.max_drawdown_pct:
            logger.warning(
                "Drawdown limit hit (%.1f%% >= %.1f%%). Rejecting trade on %s.",
                self.current_drawdown * 100,
                self.max_drawdown_pct * 100,
                decision.market_id,
            )
            return None

        amount = decision.target_amount

        # Rule 2: position size cap
        if decision.action == OrderAction.BUY:
            existing = self._position_for(decision.market_id, positions)
            current_size = existing.shares * existing.avg_price if existing else 0.0
            headroom = self.max_position_size - current_size
            if headroom <= 0:
                logger.info(
                    "Position limit reached for %s. Skipping.", decision.market_id
                )
                return None
            amount = min(amount, headroom)

        # Rule 3: total exposure cap
        if decision.action == OrderAction.BUY:
            total_exposure = sum(p.market_value for p in positions)
            exposure_headroom = self.max_total_exposure - total_exposure
            if exposure_headroom <= 0:
                logger.info("Total exposure limit reached. Skipping.")
                return None
            amount = min(amount, exposure_headroom)

        if amount <= 0:
            return None

        return Order(
            market_id=decision.market_id,
            source=decision.source,
            side=decision.side,
            action=decision.action,
            amount=round(amount, 2),
            limit_price=decision.limit_price,
        )

    @staticmethod
    def _position_for(market_id: str, positions: list[Position]) -> Position | None:
        for p in positions:
            if p.market_id == market_id:
                return p
        return None
