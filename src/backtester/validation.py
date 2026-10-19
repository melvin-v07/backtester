"""Out-of-sample validation helpers."""

from __future__ import annotations

import pandas as pd


def train_test_split(
    data: pd.DataFrame, train_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: tune on the first part, judge on the later part.

    Each slice is backtested on its own, so indicators warm up inside the slice.
    Walk-forward validation (repeat this over rolling windows) is the next step.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    cut = int(len(data) * train_fraction)
    return data.iloc[:cut], data.iloc[cut:]
