"""Download, clean and cache daily price data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw price table into the shape the engine expects.

    Result: lower-case columns, a sorted tz-naive ``date`` index with no duplicates,
    and no rows missing an open or close price.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    missing = {"open", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"price data is missing required columns: {sorted(missing)}")

    df = df[[c for c in PRICE_COLUMNS if c in df.columns]]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "close"])


def load_prices(
    ticker: str,
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data",
) -> pd.DataFrame:
    """Load split/dividend-adjusted daily prices, caching them as CSV.

    The cache key includes the date range. With ``end=None`` the cached file is not refreshed
    automatically; delete it (or pass an explicit ``end``) to re-download.
    """
    cache_path = Path(cache_dir) / f"{ticker}_{start}_{end or 'latest'}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col="date", parse_dates=["date"])

    import yfinance as yf  # imported lazily so tests never need the network

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"no price data returned for {ticker!r}")

    prices = clean_prices(raw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    return prices
