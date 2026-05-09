# US30 Liquidity Sweep Backtester

This project is an independent quant research dashboard built to explore whether a discretionary trading strategy can be converted into a rules-based research system.

The strategy idea is based on market bias, liquidity sweeps, break of structure, and retracement-based entries. The current dashboard is the first version and focuses on market data, trend bias, risk, return and visualisation.

## Current Version

The current version includes:

- Market selection across major index and macro-related assets
- Price chart with moving averages
- Moving-average trend bias
- Total return calculation
- Annualised volatility
- Maximum drawdown
- Passive equity curve
- Recent market data table

## Strategy Framework

The full version of the system will test the following process:

1. Identify higher-timeframe market bias.
2. Mark the London session high and low.
3. Detect whether price sweeps above or below those session levels.
4. Wait for break of structure after the sweep.
5. Model a 0.5 Fibonacci retracement entry.
6. Apply fixed stop-loss and take-profit rules.
7. Analyse win rate, drawdown, risk/reward and benchmark-relative performance.

## Purpose

This is a research project only. It is not financial advice or a live trading system.

The purpose is to practise:

- Python data analysis
- Financial market research
- Backtesting logic
- Dashboard development
- Strategy evaluation
- Clear communication of quantitative results

## Tools Used

- Python
- pandas
- yfinance
- Plotly
- Streamlit
- GitHub

## Next Steps

- Add intraday US30 and US500 data
- Define London session highs and lows
- Add liquidity sweep detection
- Add break of structure logic
- Add 0.5 Fibonacci retracement entries
- Build a trade log
- Add equity curve and drawdown for the strategy itself
- Deploy the dashboard online
- Link the final dashboard to the Rand & Risk website