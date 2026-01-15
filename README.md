# BOA – Treasury FX/Rates Backtesting & Risk Dashboard (Public)

Public, educational version of a Treasury research stack:
- Python backtesting for carry / momentum / mean reversion signals (cost-aware)
- Risk module: vol tracking, historical VaR/ES, basic stress tests
- Automated reporting to Excel + optional Plotly dashboard

## Setup
pip install -r requirements.txt

## Run
python examples/run_backtest.py
python examples/run_risk_report.py
python examples/run_dashboard.py

## Structure
- src/: core modules (data IO, strategies, backtester, risk, reporting)
- examples/: runnable scripts
- data/: small sample CSVs (replace with your own data)
