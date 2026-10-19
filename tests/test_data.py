import pandas as pd
import pytest

from backtester.data import clean_prices


def test_clean_prices_normalises_a_messy_frame():
    raw = pd.DataFrame(
        {
            "Open": [11.0, 10.0, None, 12.0, 99.0],
            "Close": [11.5, 10.5, 9.0, 12.5, 98.0],
            "Volume": [1, 2, 3, 4, 5],
            "Dividends": [0, 0, 0, 0, 0],
        },
        index=pd.to_datetime(
            ["2024-01-03", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-05"]
        ),
    )
    clean = clean_prices(raw)
    assert list(clean.columns) == ["open", "close", "volume"]
    assert clean.index.is_monotonic_increasing
    assert clean.index.is_unique
    assert clean.index.name == "date"
    assert pd.Timestamp("2024-01-04") not in clean.index  # open was missing
    assert clean.loc["2024-01-05", "open"] == 99.0  # duplicate date keeps the last row


def test_clean_prices_handles_multiindex_columns_and_timezones():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize("America/New_York")
    columns = pd.MultiIndex.from_tuples([("Open", "SPY"), ("Close", "SPY")])
    raw = pd.DataFrame([[10.0, 11.0], [11.0, 12.0]], index=index, columns=columns)
    clean = clean_prices(raw)
    assert list(clean.columns) == ["open", "close"]
    assert clean.index.tz is None


def test_clean_prices_requires_open_and_close():
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-02"])))
