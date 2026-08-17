"""
Runs the US30 liquidity sweep backtest and writes the results to JSON.

This replaces the Streamlit app as the source of truth for the Rand & Risk
website's Quant Lab section. Run this script whenever you want to refresh
the numbers shown on the site, then copy the output file into the
rand-and-risk repo (public/data/quant-lab.json) and redeploy.
"""

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from liquidity_sweep import (
    load_intraday_data,
    detect_short_trade_entries,
    summarise_short_trade_entries,
)

MARKET_LABEL = "US30 Proxy - Dow Futures"
TICKER = "YM=F"
LONDON_START = "08:00"
LONDON_END = "13:00"
INTRADAY_PERIODS_TO_TRY = ["60d", "30d", "15d", "7d"]

OUTPUT_PATH = "quant_lab_data.json"


def load_all_data():
    for period in INTRADAY_PERIODS_TO_TRY:
        data_15m = load_intraday_data(ticker=TICKER, period=period, interval="15m")
        data_5m = load_intraday_data(ticker=TICKER, period=period, interval="5m")

        if not data_15m.empty and not data_5m.empty:
            return data_15m, data_5m, period

    return pd.DataFrame(), pd.DataFrame(), "No data"


def main():
    data_15m, data_5m, active_period = load_all_data()

    if data_15m.empty or data_5m.empty:
        print("No intraday data was loaded from yfinance.", file=sys.stderr)
        sys.exit(1)

    short_trade_table = detect_short_trade_entries(
        data_15m=data_15m,
        data_5m=data_5m,
        london_start=LONDON_START,
        london_end=LONDON_END,
    )
    short_trade_summary = summarise_short_trade_entries(short_trade_table)

    if short_trade_table.empty:
        print("No short entry model data was available.", file=sys.stderr)
        sys.exit(1)

    total_model_days = len(short_trade_table)
    entry_triggered_count = int(short_trade_table["Entry Triggered"].sum())

    profitable_trades = int(short_trade_table["Outcome"].isin(["Win", "Session Close Profit"]).sum())
    losing_trades = int(short_trade_table["Outcome"].isin(["Loss", "Session Close Loss"]).sum())
    flat_trades = int(short_trade_table["Outcome"].isin(["Break-even", "Session Close Flat"]).sum())

    session_close_profit = int((short_trade_table["Outcome"] == "Session Close Profit").sum())
    session_close_loss = int((short_trade_table["Outcome"] == "Session Close Loss").sum())
    session_close_flat = int((short_trade_table["Outcome"] == "Session Close Flat").sum())

    no_entry = int((short_trade_table["Outcome"] == "No Entry").sum())
    no_sweep = int((short_trade_table["Outcome"] == "No Sweep").sum())
    ambiguous = int((short_trade_table["Outcome"] == "Ambiguous").sum())

    resolved_trades = profitable_trades + losing_trades + flat_trades
    model_win_rate = (profitable_trades / resolved_trades * 100) if resolved_trades > 0 else 0

    average_r = short_trade_table["R Multiple"].dropna().mean()
    total_r = short_trade_table["R Multiple"].dropna().sum()
    entry_rate = (entry_triggered_count / total_model_days * 100) if total_model_days > 0 else 0

    executed_trades = short_trade_table[
        (short_trade_table["Entry Triggered"] == True)  # noqa: E712
        & (short_trade_table["R Multiple"].notna())
    ].copy()

    equity_curve = []
    max_r_drawdown = 0
    trade_sharpe = 0

    if not executed_trades.empty:
        executed_trades["Date"] = pd.to_datetime(executed_trades["Date"])
        executed_trades = executed_trades.sort_values(["Date", "Entry Time"]).reset_index(drop=True)
        executed_trades["Trade Number"] = executed_trades.index + 1
        executed_trades["Cumulative R"] = executed_trades["R Multiple"].cumsum()
        executed_trades["Running Peak R"] = executed_trades["Cumulative R"].cummax()
        executed_trades["Drawdown R"] = executed_trades["Cumulative R"] - executed_trades["Running Peak R"]
        max_r_drawdown = float(executed_trades["Drawdown R"].min())

        r_series = executed_trades["R Multiple"].dropna()
        if len(r_series) > 1 and r_series.std() != 0:
            trade_sharpe = float((r_series.mean() / r_series.std()) * (len(r_series) ** 0.5))

        equity_curve = [
            {
                "tradeNumber": int(row["Trade Number"]),
                "cumulativeR": round(float(row["Cumulative R"]), 4),
                "runningPeakR": round(float(row["Running Peak R"]), 4),
                "drawdownR": round(float(row["Drawdown R"]), 4),
            }
            for _, row in executed_trades.iterrows()
        ]

    outcome_summary = [
        {"outcome": row["Outcome"], "count": int(row["Count"])}
        for _, row in short_trade_summary.iterrows()
    ]

    display_columns = [
        "Date", "Sweep Time", "BOS Time", "Entry Time", "London High",
        "London Low", "Sweep High", "Entry Level", "Stop Level",
        "Target Level", "Outcome", "R Multiple", "Plain English Result",
    ]
    recent_trades_df = short_trade_table[display_columns].copy()
    recent_trades_df = recent_trades_df.sort_values(
        by=["Date", "Entry Time"], ascending=[False, False]
    ).head(20)

    def clean_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, float):
            return round(value, 4)
        return str(value)

    recent_trades = [
        {col: clean_value(row[col]) for col in display_columns}
        for _, row in recent_trades_df.iterrows()
    ]

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "modelSettings": {
            "market": MARKET_LABEL,
            "londonRange": f"{LONDON_START}–{LONDON_END}",
            "dataUsed": "15m + 5m",
            "dataWindow": active_period,
        },
        "snapshot": {
            "totalModelDays": total_model_days,
            "entriesTriggered": entry_triggered_count,
            "entryRate": round(entry_rate, 2),
            "modelWinRate": round(model_win_rate, 2),
            "profitableTrades": profitable_trades,
            "losingTrades": losing_trades,
            "flatTrades": flat_trades,
            "averageR": None if pd.isna(average_r) else round(float(average_r), 2),
            "totalR": round(float(total_r), 2),
            "tradeSharpe": round(trade_sharpe, 2),
            "maxRDrawdown": round(max_r_drawdown, 2),
            "noEntryOrNoSweep": no_entry + no_sweep,
        },
        "sessionClose": {
            "profit": session_close_profit,
            "loss": session_close_loss,
            "flat": session_close_flat,
            "ambiguous": ambiguous,
        },
        "equityCurve": equity_curve,
        "outcomeSummary": outcome_summary,
        "recentTrades": recent_trades,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_PATH} ({total_model_days} model days, {entry_triggered_count} entries).")


if __name__ == "__main__":
    main()
