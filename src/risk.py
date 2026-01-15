from __future__ import annotations
import numpy as np
import pandas as pd


def realized_vol(pnl: pd.Series, window: int = 60, ann_days: int = 252) -> pd.Series:
    return np.sqrt(ann_days) * pnl.rolling(window).std(ddof=1)


def var_es_historical(pnl: pd.Series, alpha: float = 0.01) -> tuple[float, float]:
    """
    VaR/ES on PnL (loss tail). Returns positive VaR/ES (loss numbers).
    """
    loss = -pnl.dropna().values
    var = float(np.quantile(loss, 1 - (1 - alpha)))  # same as quantile(loss, alpha) for loss
    tail = loss[loss >= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def stress_shock(pnl: pd.Series, shock_mult: float = 2.0) -> pd.Series:
    """
    Simple stress: scale losses by shock_mult when pnl is negative.
    """
    s = pnl.copy()
    neg = s < 0
    s.loc[neg] = s.loc[neg] * shock_mult
    return s


def risk_report(pnl: pd.Series, alpha: float = 0.01) -> dict:
    var, es = var_es_historical(pnl, alpha=alpha)
    return {
        "VaR_hist": var,
        "ES_hist": es,
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)),
    }
