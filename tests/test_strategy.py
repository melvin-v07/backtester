import pandas as pd
import pytest

from backtester.strategy import BuyAndHold, MovingAverageCrossover


def frame(closes):
    return pd.DataFrame({"close": closes, "open": closes})


def test_buy_and_hold_is_always_fully_invested():
    assert BuyAndHold().target_position(frame([1.0, 2.0])) == 1.0


def test_crossover_is_in_cash_until_enough_history():
    assert MovingAverageCrossover(2, 4).target_position(frame([1.0, 2.0, 3.0])) == 0.0


def test_crossover_goes_long_in_an_uptrend():
    assert MovingAverageCrossover(2, 4).target_position(frame([1.0, 2.0, 3.0, 4.0])) == 1.0


def test_crossover_stays_in_cash_in_a_downtrend():
    assert MovingAverageCrossover(2, 4).target_position(frame([4.0, 3.0, 2.0, 1.0])) == 0.0


@pytest.mark.parametrize("fast,slow", [(0, 5), (5, 5), (10, 5), (-1, 5)])
def test_crossover_rejects_bad_windows(fast, slow):
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast, slow)
