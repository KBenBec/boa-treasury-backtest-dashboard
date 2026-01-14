from __future__ import annotations
import numpy as np
import pandas as pd


def zscore(x: pd.Series, window: int = 60) -> pd.Series:
    m = x.rolling(window, min_periods=max(10, window // 5)).mean()
    s = x.rolling(window, min_periods=max(10, window // 5)).std(ddof=1).replace(0, np.nan)
    return (x - m) / s


def signal_momentum(returns: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Momentum score ~ rolling sum of returns.
    """
    return returns.rolling(lookback).sum()


def signal_mean_reversion(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Mean reversion score ~ -zscore of cumulative return.
    """
    cum = returns.cumsum()
    sig = cum.apply(lambda s: -zscore(s, window=window))
    return sig


def signal_carry(carry: pd.DataFrame) -> pd.DataFrame:
    """
    Carry signal: higher carry => long.
    """
    return carry.copy()


def normalize_scores(scores: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Cross-sectional z-score per date, then clip.
    """
    def _cs_z(row: pd.Series) -> pd.Series:
        mu = row.mean()
        sd = row.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return row * 0.0
        return (row - mu) / sd

    z = scores.apply(_cs_z, axis=1).clip(-clip, clip)
    return z.fillna(0.0)
