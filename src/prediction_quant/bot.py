"""Core bot loop that orchestrates the full pipeline:
fetch markets -> generate signals -> evaluate strategy -> risk check -> execute.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from prediction_quant.execution.executor import Executor
from prediction_quant.markets.base import MarketClient
from prediction_quant.models import Market, Position
from prediction_quant.risk.manager import RiskManager
from prediction_quant.strategy.composite import CompositeSignalStrategy

logger = logging.getLogger(__name__)


class Bot:
    """Main trading bot that runs the quant pipeline in a loop."""

    def __init__(
        self,
        clients: dict[str, MarketClient],
        strategy: CompositeSignalStrategy,
        risk_manager: RiskManager,
        executor: Executor,
        poll_interval: int = 60,
        market_ids: list[str] | None = None,
        min_volume: float = 0.0,
    ) -> None:
        self._clients = clients
        self._strategy = strategy
        self._risk = risk_manager
        self._executor = executor
        self._poll_interval = poll_interval
        self._market_ids = market_ids
        self._min_volume = min_volume
        self._running = False

    async def run_once(self) -> int:
        """Execute a single pass of the pipeline. Returns number of orders placed."""
        orders_placed = 0

        for source, client in self._clients.items():
            try:
                markets = await self._fetch_markets(client)
                positions = await client.get_positions()
                logger.info(
                    "[%s] Scanning %d markets, %d open positions",
                    source,
                    len(markets),
                    len(positions),
                )
                self._update_portfolio_value(positions)

                for market in markets:
                    order_id = await self._process_market(client, market, positions)
                    if order_id:
                        orders_placed += 1

            except Exception:
                logger.exception("Error processing source %s", source)

        return orders_placed

    async def run(self) -> None:
        """Run the bot in a continuous loop."""
        self._running = True
        logger.info("Bot started — polling every %ds", self._poll_interval)
        while self._running:
            ts = datetime.now(UTC).isoformat()
            logger.info("--- Tick at %s ---", ts)
            try:
                n = await self.run_once()
                logger.info("Placed %d orders this tick.", n)
            except Exception:
                logger.exception("Unhandled error in bot loop")
            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False

    # -- internals -------------------------------------------------------------

    async def _fetch_markets(self, client: MarketClient) -> list[Market]:
        if self._market_ids:
            return await client.get_markets_batch(self._market_ids)
        return await client.list_markets(
            limit=30,
            active_only=True,
            min_volume=self._min_volume,
        )

    async def _process_market(
        self,
        client: MarketClient,
        market: Market,
        positions: list[Position],
    ) -> str | None:
        try:
            history = await client.get_price_history(market.id)
        except Exception:
            logger.warning("Could not fetch history for %s", market.id)
            return None

        signals = self._strategy.compute_signals(market, history)
        if not signals:
            return None

        position = next(
            (p for p in positions if p.market_id == market.id), None
        )
        decision = self._strategy.evaluate(market, history, signals, position)
        if decision is None:
            return None

        order = self._risk.check(decision, positions)
        if order is None:
            return None

        logger.info(
            "Decision: %s | %s",
            decision.reason,
            f"{order.action.value} {order.side.value} ${order.amount:.2f}",
        )
        return await self._executor.execute(order)

    def _update_portfolio_value(self, positions: list[Position]) -> None:
        total = sum(p.market_value for p in positions)
        self._risk.update_portfolio_value(total)
