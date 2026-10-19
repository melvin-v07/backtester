import numpy as np
import pandas as pd
import pytest


def _make_prices(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLCV data: a random walk with a small upward drift."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


@pytest.fixture
def make_prices():
    return _make_prices


@pytest.fixture
def prices() -> pd.DataFrame:
    return _make_prices()
