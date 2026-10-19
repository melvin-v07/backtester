"""Cash, holdings and order execution (long-only, single asset)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BPS = 1e4


@dataclass(frozen=True)
class Trade:
    """One executed order. ``shares`` is positive for a buy and negative for a sell."""

    date: pd.Timestamp
    shares: float
    price: float  # fill price, after slippage
    commission: float


class Portfolio:
    def __init__(self, cash: float, commission_bps: float = 0.0, slippage_bps: float = 0.0) -> None:
        if cash <= 0:
            raise ValueError("starting cash must be positive")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        self.cash = float(cash)
        self.shares = 0.0
        self.commission_rate = commission_bps / BPS
        self.slippage_rate = slippage_bps / BPS
        self.trades: list[Trade] = []

    def value(self, price: float) -> float:
        """Total portfolio value if the asset is marked at ``price``."""
        return self.cash + self.shares * price

    def rebalance_to(self, target_weight: float, price: float, date: pd.Timestamp) -> Trade | None:
        """Trade so that ``target_weight`` of portfolio value is held in the asset.

        ``price`` is the quoted price (the bar's open). Buys fill slightly above it and sells
        slightly below it (slippage), and commission is charged on the traded value.
        Returns the trade, or ``None`` if the position is already at target.
        """
        if not 0.0 <= target_weight <= 1.0:
            raise ValueError(f"target weight must be between 0 and 1, got {target_weight}")

        total = self.value(price)
        delta_value = target_weight * total - self.shares * price
        if abs(delta_value) < 1e-9 * max(total, 1.0):
            return None

        if delta_value > 0:  # buy: spend delta_value including commission
            fill = price * (1 + self.slippage_rate)
            qty = delta_value / (fill * (1 + self.commission_rate))
        else:  # sell: never sell more than we hold
            fill = price * (1 - self.slippage_rate)
            qty = max(delta_value / fill, -self.shares)

        commission = abs(qty) * fill * self.commission_rate
        self.cash -= qty * fill + commission
        self.shares += qty
        if abs(self.shares) < 1e-12:
            self.shares = 0.0

        trade = Trade(date=date, shares=qty, price=fill, commission=commission)
        self.trades.append(trade)
        return trade
