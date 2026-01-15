import numpy as np
import pandas as pd

from src.io_data import load_prices
from src.risk import realized_vol, risk_report, stress_shock


def main():
    prices = load_prices("data/sample_prices.csv")
    rets = np.log(prices.clip(lower=1e-12)).diff().dropna()

    # Example: equal-weight pnl proxy
    pnl = rets.mean(axis=1)

    print("=== Risk report (base) ===")
    print(risk_report(pnl, alpha=0.01))

    print("\n=== Risk report (stressed) ===")
    pnl_s = stress_shock(pnl, shock_mult=2.0)
    print(risk_report(pnl_s, alpha=0.01))

    vol = realized_vol(pnl, window=60)
    print("\nRealized vol (last):", float(vol.dropna().iloc[-1]))


if __name__ == "__main__":
    main()
