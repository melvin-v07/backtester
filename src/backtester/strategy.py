"""Trading strategies.

A strategy never trades directly. Each day it is shown the history so far and answers one
question: what fraction of the portfolio should be invested tomorrow? The engine decides when
and at what price that becomes a trade, which keeps look-ahead bias out of strategy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all strategies."""

    @abstractmethod
    def target_position(self, history: pd.DataFrame) -> float:
        """Return the desired fraction of portfolio value held in the asset, from 0.0 to 1.0.

        ``history`` holds every bar up to and including today's close, and nothing later.
        """


class BuyAndHold(Strategy):
    """Always fully invested. The benchmark every other strategy is compared with."""

    def target_position(self, history: pd.DataFrame) -> float:
        return 1.0


class MovingAverageCrossover(Strategy):
    """Fully invested while the fast moving average is above the slow one, otherwise in cash."""

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast <= 0 or slow <= 0:
            raise ValueError("moving-average windows must be positive")
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow

    def target_position(self, history: pd.DataFrame) -> float:
        if len(history) < self.slow:
            return 0.0  # not enough data yet to compute the slow average
        close = history["close"]
        fast_ma = close.iloc[-self.fast :].mean()
        slow_ma = close.iloc[-self.slow :].mean()
        return 1.0 if fast_ma > slow_ma else 0.0
