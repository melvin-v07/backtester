import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from backtester.engine import run_backtest  # noqa: E402
from backtester.plots import plot_equity_curves  # noqa: E402
from backtester.strategy import BuyAndHold  # noqa: E402


def test_plot_writes_a_file(prices, tmp_path):
    equity = run_backtest(prices, BuyAndHold()).equity
    out = tmp_path / "nested" / "chart.png"
    fig = plot_equity_curves({"Buy and hold": equity}, path=out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0
