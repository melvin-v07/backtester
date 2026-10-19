"""A small event-driven backtesting engine."""

from backtester.engine import BacktestResult, run_backtest
from backtester.strategy import BuyAndHold, MovingAverageCrossover, Strategy

__all__ = [
    "BacktestResult",
    "BuyAndHold",
    "MovingAverageCrossover",
    "Strategy",
    "run_backtest",
]
