"""Charts for backtest results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_equity_curves(
    curves: Mapping[str, pd.Series],
    path: str | Path | None = None,
    title: str = "Equity curve",
) -> Figure:
    """Plot portfolio value (top) and drawdown (bottom) for one or more named equity curves."""
    fig, (ax_equity, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for name, equity in curves.items():
        ax_equity.plot(equity.index, equity.to_numpy(), label=name)
        drawdown = equity / equity.cummax() - 1
        ax_dd.plot(drawdown.index, drawdown.to_numpy(), label=name)

    ax_equity.set_title(title)
    ax_equity.set_ylabel("Portfolio value")
    ax_equity.legend()
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.tight_layout()

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
