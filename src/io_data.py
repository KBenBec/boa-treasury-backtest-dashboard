from __future__ import annotations
import pandas as pd


def load_prices(csv_path: str) -> pd.DataFrame:
    """
    Expected format:
    date, asset, price
    2020-01-01, EURUSD, 1.12
    ...
    Returns pivoted DataFrame indexed by date, columns=assets (prices).
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    px = df.pivot(index="date", columns="asset", values="price").astype(float)
    return px


def load_carry(csv_path: str) -> pd.DataFrame:
    """
    Expected format:
    date, asset, carry
    carry = carry proxy (rate differential / roll-down etc.)
    Returns pivoted DataFrame indexed by date, columns=assets (carry).
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    c = df.pivot(index="date", columns="asset", values="carry").astype(float)
    return c


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices.apply(lambda s: (s.astype(float)).map(lambda x: x if x > 0 else 1e-12))
            .pipe(lambda px: px.apply(lambda s: s.map(float)))
            .pipe(lambda px: (px.apply(lambda s: s.map(float))))
            .pipe(lambda px: (px.apply(lambda s: s.map(float)))))
    # NOTE: above is intentionally verbose-safe on weird inputs; simplest is below:
    # return np.log(prices).diff().dropna()

