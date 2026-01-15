import numpy as np
import pandas as pd

from src.io_data import load_prices, load_carry
from src.strategies import signal_carry, signal_momentum, signal_mean_reversion, normalize_scores
from src.backtester import BacktestConfig, run_backtest
from src.reporting import export_excel, plot_equity


def main():
    prices = load_prices("data/sample_prices.csv")
    carry = load_carry("data/sample_carry.csv")
    rets = np.log(prices.clip(lower=1e-12)).diff().dropna()

    s1 = normalize_scores(signal_carry(carry.reindex(rets.index).fillna(0.0)))
    s2 = normalize_scores(signal_momentum(rets, lookback=20))
    s3 = normalize_scores(signal_mean_reversion(rets, window=60))
    score = 0.4 * s1 + 0.3 * s2 + 0.3 * s3

    cfg = BacktestConfig(vol_target_ann=0.10, cost_bps=0.5, max_gross=2.0)
    bt = run_backtest(rets, score, cfg)

    export_excel(bt, path="reports/boa_report.xlsx")

    # Optional interactive plot
    try:
        fig = plot_equity(bt)
        fig.write_html("reports/equity_curve.html")
        print("Saved dashboard: reports/equity_curve.html")
    except Exception as e:
        print("Plotly not available; skipped HTML dashboard. Error:", e)


if __name__ == "__main__":
    main()
