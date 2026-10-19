import pytest

from backtester.engine import run_backtest
from backtester.strategy import BuyAndHold, MovingAverageCrossover, Strategy


def test_orders_fill_at_next_bars_open(prices):
    result = run_backtest(prices, BuyAndHold(), commission_bps=0.0)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.date == prices.index[1]  # decided on day 0's close, filled on day 1
    assert trade.price == pytest.approx(prices["open"].iloc[1])


def test_equity_starts_at_initial_cash(prices):
    result = run_backtest(prices, BuyAndHold(), initial_cash=5_000, commission_bps=0.0)
    assert result.equity.iloc[0] == pytest.approx(5_000)


def test_buy_and_hold_matches_hand_calculation(prices):
    result = run_backtest(prices, BuyAndHold(), initial_cash=10_000, commission_bps=0.0)
    shares = 10_000 / prices["open"].iloc[1]
    assert result.equity.iloc[-1] == pytest.approx(shares * prices["close"].iloc[-1])


def test_costs_reduce_final_equity(prices):
    strategy = MovingAverageCrossover(5, 20)
    free = run_backtest(prices, strategy, commission_bps=0.0)
    costly = run_backtest(prices, strategy, commission_bps=50.0)
    assert len(free.trades) > 0
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


def test_strategy_that_never_changes_its_mind_trades_once(prices):
    class AlwaysHalf(Strategy):
        def target_position(self, history):
            return 0.5

    assert len(run_backtest(prices, AlwaysHalf(), commission_bps=0.0).trades) == 1


def test_out_of_range_target_is_rejected(prices):
    class Greedy(Strategy):
        def target_position(self, history):
            return 2.0

    with pytest.raises(ValueError):
        run_backtest(prices, Greedy())


def test_missing_columns_are_rejected(prices):
    with pytest.raises(ValueError):
        run_backtest(prices.drop(columns=["open"]), BuyAndHold())


def test_needs_at_least_two_bars(prices):
    with pytest.raises(ValueError):
        run_backtest(prices.iloc[:1], BuyAndHold())
