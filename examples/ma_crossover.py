"""Run a moving-average crossover against buy-and-hold on one ticker.

Usage:
    python examples/ma_crossover.py SPY --start 2010-01-01
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from backtester.data import load_prices
from backtester.engine import run_backtest
from backtester.metrics import summary
from backtester.plots import plot_equity_curves
from backtester.strategy import BuyAndHold, MovingAverageCrossover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--out", default="docs/ma_crossover.png")
    args = parser.parse_args()

    prices = load_prices(args.ticker, args.start, args.end)
    strategies = {
        "Buy and hold": BuyAndHold(),
        f"MA crossover ({args.fast}/{args.slow})": MovingAverageCrossover(args.fast, args.slow),
    }
    results = {
        name: run_backtest(
            prices,
            strategy,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
        )
        for name, strategy in strategies.items()
    }

    table = pd.DataFrame(
        {name: summary(r.equity, n_trades=len(r.trades)) for name, r in results.items()}
    ).T
    table["n_trades"] = table["n_trades"].astype(int)
    print(table.to_string(float_format=lambda x: f"{x:,.3f}"))

    plot_equity_curves(
        {name: r.equity for name, r in results.items()},
        path=args.out,
        title=f"{args.ticker}: strategy vs buy and hold",
    )
    plt.close("all")
    print(f"\nSaved chart to {args.out}")


if __name__ == "__main__":
    main()
