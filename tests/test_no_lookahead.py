"""Tests that a strategy cannot use information from the future.

Two checks:
  1. The history passed to the strategy on bar i ends at bar i.
  2. Rewriting every price after a cutoff date leaves results up to the cutoff unchanged.
A deliberately cheating strategy proves check 2 would catch a leak.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from backtester.engine import run_backtest
from backtester.strategy import MovingAverageCrossover, Strategy

PRICE_FIELDS = ["open", "high", "low", "close"]


class RecordingStrategy(Strategy):
    def __init__(self):
        self.last_dates = []

    def target_position(self, history):
        self.last_dates.append(history.index[-1])
        return 1.0


class LeakyStrategy(Strategy):
    """Cheats: buys if the price five bars from now is higher than today's."""

    def __init__(self, full: pd.DataFrame):
        self.full = full

    def target_position(self, history):
        today = len(history) - 1
        ahead = min(today + 5, len(self.full) - 1)
        return 1.0 if self.full["close"].iloc[ahead] > self.full["close"].iloc[today] else 0.0


def poison_future(data: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Multiply every price after position ``cutoff`` by 5."""
    out = data.copy()
    out.loc[out.index[cutoff + 1 :], PRICE_FIELDS] *= 5.0
    return out


def equity_up_to_cutoff(make_strategy, data, cutoff):
    clean = run_backtest(data, make_strategy(data), commission_bps=0.0)
    poisoned_data = poison_future(data, cutoff)
    poisoned = run_backtest(poisoned_data, make_strategy(poisoned_data), commission_bps=0.0)
    return clean, clean.equity.iloc[: cutoff + 1], poisoned.equity.iloc[: cutoff + 1]


def test_strategy_only_sees_history_up_to_today(prices):
    strategy = RecordingStrategy()
    run_backtest(prices, strategy)
    assert strategy.last_dates == list(prices.index)


def test_future_prices_cannot_change_past_results(prices):
    clean, before, after = equity_up_to_cutoff(lambda d: MovingAverageCrossover(5, 20), prices, 150)
    assert len(clean.trades) > 0  # the check is meaningless if nothing traded
    assert_series_equal(before, after)


def test_the_check_catches_a_leaky_strategy():
    n = 100
    falling = pd.DataFrame(
        {"open": np.linspace(200, 100, n), "close": np.linspace(200, 100, n)},
        index=pd.bdate_range("2020-01-01", periods=n, name="date"),
    )
    falling["high"] = falling["low"] = falling["close"]
    _, before, after = equity_up_to_cutoff(LeakyStrategy, falling, 90)
    with pytest.raises(AssertionError):
        assert_series_equal(before, after)
