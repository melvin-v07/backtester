import math

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    annualised_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    summary,
    total_return,
)


def test_total_return():
    assert total_return(pd.Series([100.0, 150.0])) == pytest.approx(0.5)


def test_cagr_of_one_year_doubling():
    equity = pd.Series(100 * 2 ** (np.arange(253) / 252))  # doubles over exactly 252 steps
    assert cagr(equity) == pytest.approx(1.0)


def test_cagr_of_flat_series_is_zero():
    assert cagr(pd.Series([100.0] * 50)) == pytest.approx(0.0)


def test_max_drawdown_known_values():
    # peaks at 120, falls to 60 (-50%), recovers to 90
    assert max_drawdown(pd.Series([100.0, 120.0, 60.0, 90.0])) == pytest.approx(-0.5)


def test_max_drawdown_of_rising_series_is_zero():
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0


def test_sharpe_known_value():
    # mean 0.01, sample std 0.01 * sqrt(2) -> daily ratio 1/sqrt(2); annualised x sqrt(252)
    returns = pd.Series([0.02, 0.0])
    assert sharpe_ratio(returns) == pytest.approx(math.sqrt(126))


def test_sharpe_is_nan_when_returns_are_constant():
    assert math.isnan(sharpe_ratio(pd.Series([0.01] * 20)))


def test_sharpe_falls_when_risk_free_rate_rises():
    returns = pd.Series([0.02, 0.0, 0.01, -0.005, 0.015])
    assert sharpe_ratio(returns, risk_free_rate=0.05) < sharpe_ratio(returns)


def test_volatility_scales_with_sqrt_of_periods():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    daily = returns.std(ddof=1)
    assert annualised_volatility(returns) == pytest.approx(daily * math.sqrt(252))


def test_summary_contains_headline_metrics():
    equity = pd.Series([100.0, 101.0, 99.0, 103.0, 104.0])
    result = summary(equity, n_trades=3)
    assert set(result) == {
        "total_return",
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "n_trades",
    }
    assert result["n_trades"] == 3
