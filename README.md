# backtester

An event-driven backtesting engine for systematic trading strategies, written in Python.

It replays historical prices one bar at a time, shows a strategy only what was knowable at that
moment, fills its orders on the **next** bar with commission and slippage, and reports performance
metrics next to a buy-and-hold benchmark.

> Status: early scaffold. The engine, a first strategy, metrics and tests work; see the roadmap.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest
python examples/ma_crossover.py SPY --start 2010-01-01
```

The example downloads prices with yfinance (cached under `data/`), runs a moving-average crossover
and buy-and-hold through the same engine, prints a metrics table and saves a chart to `docs/`.

## Design rules

1. A strategy sees history up to and including today's close, and nothing later.
2. A decision made on day N is filled at day N+1's open. This is what prevents look-ahead bias.
3. Costs are on by default (5 bps commission); slippage is configurable.
4. Every strategy is judged against buy-and-hold run through the same engine and cost model.

## Layout

```text
src/backtester/
  data.py        download, clean and cache prices
  strategy.py    Strategy base class, BuyAndHold, MovingAverageCrossover
  portfolio.py   cash, shares, order execution, costs
  engine.py      the bar-by-bar loop
  metrics.py     CAGR, volatility, Sharpe, max drawdown
  validation.py  train/test split (walk-forward comes next)
  plots.py       equity curve and drawdown chart
tests/           pytest suite, including a no-look-ahead test
examples/        runnable scripts
```

## Roadmap

- [x] Data loader with local cache
- [x] Event-driven engine with next-bar fills and costs
- [x] Core metrics
- [x] No-look-ahead tests
- [ ] Mean-reversion and momentum strategies
- [ ] Walk-forward validation
- [ ] Multi-asset comparison table
- [ ] Results section with charts (add after running real data)

## Limitations

- Long-only, one asset at a time, no leverage.
- Simple proportional cost model; no market impact or partial fills.
- Free yfinance data is adjusted and can be revised; individual stocks suffer survivorship bias.
- A backtest shows how an idea would have fared in the past, not how it will fare in future.
