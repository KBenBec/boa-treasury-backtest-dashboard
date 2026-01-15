from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    vol_target_ann: float = 0.10       # 10% annual target vol
    rebalance_freq: int = 1            # daily
    cost_bps: float = 0.5              # per unit turnover
    max_gross: float = 2.0             # leverage cap (sum |w|)
    vol_lookback: int = 60
    ann_days: int = 252


def ewma_vol(x: pd.Series, halflife: int = 20) -> pd.Series:
    lam = 0.5 ** (1.0 / max(halflife, 1))
    s2 = x.pow(2).ewm(alpha=(1 - lam), adjust=False).mean()
    return np.sqrt(s2).replace(0, np.nan).bfill()


def vol_target_weights(scores: pd.DataFrame, returns: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Convert scores to portfolio weights with:
    - cross-sectional normalization already done upstream
    - volatility targeting using portfolio vol proxy (EWMA of pnl)
    - gross leverage cap
    """
    w = scores.copy()

    # Normalize gross exposure each day
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)

    # Scale to reach target vol using rolling estimate on equal-weight pnl proxy
    pnl_proxy = (w.shift(1) * returns).sum(axis=1).fillna(0.0)
    vol = pnl_proxy.rolling(cfg.vol_lookback).std(ddof=1).replace(0, np.nan).bfill()
    daily_target = cfg.vol_target_ann / np.sqrt(cfg.ann_days)
    scale = (daily_target / vol).clip(0.0, 10.0)
    w = w.mul(scale, axis=0)

    # Gross leverage cap
    gross2 = w.abs().sum(axis=1)
    cap = (cfg.max_gross / gross2).clip(upper=1.0)
    w = w.mul(cap, axis=0)
    return w


def run_backtest(returns: pd.DataFrame, scores: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Backtest with transaction costs:
    cost = cost_bps * turnover, turnover = sum |Δw|
    """
    returns = returns.loc[scores.index].copy()
    w = vol_target_weights(scores, returns, cfg)

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = (cfg.cost_bps * 1e-4) * turnover

    pnl_gross = (w.shift(1) * returns).sum(axis=1).fillna(0.0)
    pnl_net = pnl_gross - costs

    eq = (1.0 + pnl_net).cumprod()

    out = pd.DataFrame({
        "pnl_gross": pnl_gross,
        "pnl_net": pnl_net,
        "turnover": turnover,
        "costs": costs,
        "equity": eq,
    }, index=returns.index)

    return out


def perf_metrics(pnl: pd.Series, equity: pd.Series, ann_days: int = 252) -> dict:
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    sharpe = float(np.sqrt(ann_days) * mu / (sd + 1e-12))

    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    mdd = float(dd.min())

    total_ret = float(equity.iloc[-1] - 1.0)
    ann_ret = float(equity.iloc[-1] ** (ann_days / max(len(equity), 1)) - 1.0)

    return {"Sharpe": sharpe, "MaxDrawdown": mdd, "TotalReturn": total_ret, "AnnReturn": ann_ret}
