"""CLI entry point for the prediction-quant bot."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from prediction_quant.bot import Bot
from prediction_quant.config import load_bot_settings, load_market_settings
from prediction_quant.execution.executor import Executor
from prediction_quant.markets.base import MarketClient
from prediction_quant.markets.kalshi import KalshiClient
from prediction_quant.markets.manifold import ManifoldClient
from prediction_quant.markets.polymarket import PolymarketClient
from prediction_quant.markets.robinhood import RobinhoodClient
from prediction_quant.risk.manager import RiskManager
from prediction_quant.signals.ema import EMACrossoverSignal
from prediction_quant.signals.fair_value import FairValueDivergenceSignal
from prediction_quant.signals.volume import VolumeSignal
from prediction_quant.strategy.composite import CompositeSignalStrategy


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_clients(market_cfg) -> dict[str, MarketClient]:
    clients: dict[str, MarketClient] = {}
    if market_cfg.polymarket_api_key:
        clients["polymarket"] = PolymarketClient(
            api_key=market_cfg.polymarket_api_key,
            api_secret=market_cfg.polymarket_api_secret,
            api_passphrase=market_cfg.polymarket_api_passphrase,
        )
    if market_cfg.manifold_api_key:
        clients["manifold"] = ManifoldClient(api_key=market_cfg.manifold_api_key)
    if market_cfg.kalshi_api_key:
        clients["kalshi"] = KalshiClient(
            api_key=market_cfg.kalshi_api_key,
            private_key_path=market_cfg.kalshi_private_key_path,
            demo=market_cfg.kalshi_demo,
        )
    if market_cfg.robinhood_access_token:
        clients["robinhood"] = RobinhoodClient(
            access_token=market_cfg.robinhood_access_token,
        )
    return clients


@click.group()
def cli() -> None:
    """prediction-quant: quantitative trading bot for prediction markets."""


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--market-id",
    "-m",
    multiple=True,
    help="Specific market IDs to trade. Can be repeated.",
)
@click.option("--once", is_flag=True, help="Run a single tick then exit.")
@click.option(
    "--source",
    type=click.Choice(["polymarket", "manifold", "kalshi", "robinhood", "all"]),
    default="all",
)
def run(verbose: bool, market_id: tuple[str, ...], once: bool, source: str) -> None:
    """Start the trading bot."""
    _setup_logging(verbose)
    market_cfg = load_market_settings()
    bot_cfg = load_bot_settings()

    clients = _build_clients(market_cfg)
    if source != "all":
        clients = {k: v for k, v in clients.items() if k == source}

    if not clients:
        click.echo(
            "No market clients configured. Set API keys in .env (see .env.example).",
            err=True,
        )
        raise SystemExit(1)

    strategy = CompositeSignalStrategy(
        generators=[
            EMACrossoverSignal(),
            VolumeSignal(),
            FairValueDivergenceSignal(),
        ],
        base_amount=bot_cfg.max_position_size / 10,
    )
    risk_mgr = RiskManager(
        max_position_size=bot_cfg.max_position_size,
        max_total_exposure=bot_cfg.max_total_exposure,
        max_drawdown_pct=bot_cfg.max_drawdown_pct,
    )
    executor = Executor(clients=clients, dry_run=bot_cfg.dry_run)

    bot = Bot(
        clients=clients,
        strategy=strategy,
        risk_manager=risk_mgr,
        executor=executor,
        poll_interval=bot_cfg.poll_interval_seconds,
        market_ids=list(market_id) if market_id else None,
    )

    if once:
        n = asyncio.run(bot.run_once())
        click.echo(f"Single tick complete. {n} order(s) placed.")
    else:
        click.echo(
            f"Starting bot (dry_run={bot_cfg.dry_run}, "
            f"poll={bot_cfg.poll_interval_seconds}s, "
            f"sources={list(clients.keys())})"
        )
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            click.echo("\nBot stopped.")


@cli.command()
def scan() -> None:
    """Scan markets and print top opportunities (no trading)."""
    _setup_logging(verbose=False)
    market_cfg = load_market_settings()
    clients = _build_clients(market_cfg)

    if not clients:
        click.echo("No market clients configured.", err=True)
        raise SystemExit(1)

    strategy = CompositeSignalStrategy(
        generators=[
            EMACrossoverSignal(),
            VolumeSignal(),
            FairValueDivergenceSignal(),
        ],
    )

    async def _scan() -> None:
        for source, client in clients.items():
            async with client:
                markets = await client.list_markets(limit=20, active_only=True)
                click.echo(f"\n=== {source.upper()} ({len(markets)} markets) ===")
                for market in markets:
                    history = await client.get_price_history(market.id)
                    signals = strategy.compute_signals(market, history)
                    if not signals:
                        continue
                    total_w = sum(s.confidence for s in signals)
                    if total_w == 0:
                        continue
                    agg = sum(s.value * s.confidence for s in signals) / total_w
                    if abs(agg) < 0.15:
                        continue
                    direction = "YES" if agg > 0 else "NO"
                    click.echo(
                        f"  [{direction} {abs(agg):.2f}] {market.question[:80]}"
                        f"  (p={market.probability:.2f}, vol24h=${market.volume_24h:,.0f})"
                    )

    asyncio.run(_scan())


if __name__ == "__main__":
    cli()
