from __future__ import annotations
import pandas as pd

try:
    import plotly.express as px
except Exception:
    px = None


def export_excel(backtest_df: pd.DataFrame, path: str = "reports/boa_report.xlsx") -> None:
    """
    Export backtest time series to Excel (simple recruiter-friendly output).
    """
    # create folder if needed
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        backtest_df.to_excel(w, sheet_name="backtest")
    print(f"Saved Excel report: {path}")


def plot_equity(backtest_df: pd.DataFrame):
    """
    Optional Plotly equity plot.
    """
    if px is None:
        raise ImportError("plotly not installed. Add it to requirements or skip dashboard.")
    fig = px.line(backtest_df.reset_index(), x=backtest_df.index.name or "index", y="equity", title="Equity Curve")
    return fig
