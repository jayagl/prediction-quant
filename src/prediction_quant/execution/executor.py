"""Order execution layer.

Routes approved orders to the correct market client and handles
dry-run mode for paper trading.
"""

from __future__ import annotations

import logging

from prediction_quant.markets.base import MarketClient
from prediction_quant.models import Order

logger = logging.getLogger(__name__)


class Executor:
    """Dispatches orders to market clients."""

    def __init__(
        self,
        clients: dict[str, MarketClient],
        dry_run: bool = True,
    ) -> None:
        self._clients = clients
        self.dry_run = dry_run
        self.order_log: list[tuple[Order, str]] = []  # (order, result)

    async def execute(self, order: Order) -> str:
        """Execute an order. Returns the order ID or a dry-run tag."""
        if self.dry_run:
            tag = f"DRY-RUN-{len(self.order_log)}"
            logger.info(
                "[DRY RUN] %s %s %.2f on %s (%s) limit=%s",
                order.action.value,
                order.side.value,
                order.amount,
                order.market_id,
                order.source,
                order.limit_price,
            )
            self.order_log.append((order, tag))
            return tag

        client = self._clients.get(order.source)
        if client is None:
            raise ValueError(f"No client registered for source '{order.source}'")

        order_id = await client.place_order(order)
        logger.info(
            "Placed order %s: %s %s %.2f on %s (%s)",
            order_id,
            order.action.value,
            order.side.value,
            order.amount,
            order.market_id,
            order.source,
        )
        self.order_log.append((order, order_id))
        return order_id
