"""Performance metrics computed from an equity curve or a returns series."""

from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate: the smooth yearly return that gives the same end value."""
    periods = len(equity) - 1
    if periods <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return math.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / periods) - 1)


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised mean excess return per unit of volatility. ``risk_free_rate`` is annual."""
    excess = returns - risk_free_rate / periods_per_year
    spread = excess.std(ddof=1)
    if math.isnan(spread) or spread < 1e-12:
        return math.nan
    return float(excess.mean() / spread * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative fraction (-0.35 means a 35% fall)."""
    return float((equity / equity.cummax() - 1).min())


def summary(equity: pd.Series, n_trades: int | None = None) -> dict[str, float]:
    """The headline numbers for one backtest."""
    returns = equity.pct_change().dropna()
    result = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualised_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
    }
    if n_trades is not None:
        result["n_trades"] = float(n_trades)
    return result
