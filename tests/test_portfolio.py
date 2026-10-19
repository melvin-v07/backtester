import pandas as pd
import pytest

from backtester.portfolio import Portfolio

DATE = pd.Timestamp("2024-01-02")


def test_full_buy_with_no_costs():
    p = Portfolio(cash=1000)
    trade = p.rebalance_to(1.0, price=100, date=DATE)
    assert trade is not None
    assert p.shares == pytest.approx(10)
    assert p.cash == pytest.approx(0)
    assert trade.price == pytest.approx(100)


def test_commission_reduces_shares_bought():
    p = Portfolio(cash=1000, commission_bps=100)  # 1%
    trade = p.rebalance_to(1.0, price=100, date=DATE)
    assert p.shares == pytest.approx(1000 / 101)
    assert trade is not None
    assert trade.commission == pytest.approx(p.shares * 100 * 0.01)
    assert p.cash == pytest.approx(0, abs=1e-9)


def test_slippage_makes_buys_fill_higher():
    p = Portfolio(cash=1000, slippage_bps=100)
    trade = p.rebalance_to(1.0, price=100, date=DATE)
    assert trade is not None
    assert trade.price == pytest.approx(101)
    assert p.shares == pytest.approx(1000 / 101)


def test_slippage_makes_sells_fill_lower():
    p = Portfolio(cash=1000, slippage_bps=100)
    p.rebalance_to(1.0, price=100, date=DATE)
    trade = p.rebalance_to(0.0, price=100, date=DATE)
    assert trade is not None
    assert trade.price == pytest.approx(99)
    assert p.shares == 0


def test_round_trip_with_no_costs_captures_the_price_move():
    p = Portfolio(cash=1000)
    p.rebalance_to(1.0, price=100, date=DATE)
    p.rebalance_to(0.0, price=110, date=DATE)
    assert p.shares == 0
    assert p.cash == pytest.approx(1100)


def test_partial_target():
    p = Portfolio(cash=1000)
    p.rebalance_to(0.5, price=100, date=DATE)
    assert p.shares == pytest.approx(5)
    assert p.cash == pytest.approx(500)


def test_no_trade_when_already_at_target():
    p = Portfolio(cash=1000)
    p.rebalance_to(1.0, price=100, date=DATE)
    assert p.rebalance_to(1.0, price=100, date=DATE) is None
    assert len(p.trades) == 1


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_invalid_target_is_rejected(bad):
    with pytest.raises(ValueError):
        Portfolio(cash=1000).rebalance_to(bad, price=100, date=DATE)


def test_cash_never_goes_negative_with_costs():
    p = Portfolio(cash=1000, commission_bps=50, slippage_bps=50)
    for price in (100, 120, 90, 130):
        p.rebalance_to(1.0, price=price, date=DATE)
        p.rebalance_to(0.0, price=price, date=DATE)
        assert p.cash > 0
        assert p.shares == 0
