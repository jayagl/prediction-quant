"""Tests for the executor."""

import pytest

from prediction_quant.execution.executor import Executor
from prediction_quant.models import Order, OrderAction, Side


class TestExecutor:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_client(self):
        executor = Executor(clients={}, dry_run=True)
        order = Order(
            market_id="m1",
            source="test",
            side=Side.YES,
            action=OrderAction.BUY,
            amount=10.0,
        )
        result = await executor.execute(order)
        assert result.startswith("DRY-RUN")
        assert len(executor.order_log) == 1

    @pytest.mark.asyncio
    async def test_missing_client_raises(self):
        executor = Executor(clients={}, dry_run=False)
        order = Order(
            market_id="m1",
            source="unknown",
            side=Side.YES,
            action=OrderAction.BUY,
            amount=10.0,
        )
        with pytest.raises(ValueError, match="No client"):
            await executor.execute(order)
