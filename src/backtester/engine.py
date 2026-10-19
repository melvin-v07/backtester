"""The bar-by-bar backtest loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.portfolio import Portfolio, Trade
from backtester.strategy import Strategy


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series  # portfolio value at each bar's close
    trades: list[Trade]

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_cash: float = 10_000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Replay ``data`` through ``strategy``.

    Timeline for each bar i:
      1. Fill the order decided at the end of bar i-1, at bar i's open.
      2. Record portfolio value at bar i's close.
      3. Show the strategy history up to and including bar i; its answer becomes the order
         for bar i+1.
    A strategy therefore can never trade on information from the future.
    """
    missing = {"open", "close"} - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {sorted(missing)}")
    if len(data) < 2:
        raise ValueError("need at least two bars to run a backtest")

    portfolio = Portfolio(initial_cash, commission_bps, slippage_bps)
    opens = data["open"].to_numpy(dtype=float)
    closes = data["close"].to_numpy(dtype=float)
    equity = np.empty(len(data))

    pending: float | None = None
    last_target: float | None = None

    for i in range(len(data)):
        if pending is not None:
            portfolio.rebalance_to(pending, opens[i], data.index[i])
            pending = None

        equity[i] = portfolio.value(closes[i])

        target = float(strategy.target_position(data.iloc[: i + 1]))
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"strategy returned target {target} on {data.index[i]}; need 0 to 1")
        if target != last_target:  # only trade when the strategy changes its mind
            pending = target
            last_target = target

    return BacktestResult(
        equity=pd.Series(equity, index=data.index, name="equity"),
        trades=portfolio.trades,
    )
