import pytest

from backtester.validation import train_test_split


def test_split_is_chronological_and_complete(prices):
    train, test = train_test_split(prices, 0.7)
    assert len(train) == 140
    assert len(test) == 60
    assert train.index.max() < test.index.min()


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_split_rejects_bad_fraction(prices, bad):
    with pytest.raises(ValueError):
        train_test_split(prices, bad)
