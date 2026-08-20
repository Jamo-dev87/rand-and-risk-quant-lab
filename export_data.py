"""
Runs the US30 liquidity sweep backtest (both the short and long models) and
writes the combined results to JSON.

This replaces the Streamlit app as the source of truth for the Rand & Risk
website's Quant Lab section. Run this script whenever you want to refresh
the numbers shown on the site, then copy the output file into the
rand-and-risk repo (src/data/quantLabData.json) and redeploy.
"""

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from liquidity_sweep import (
    load_intraday_data,
    detect_short_trade_entries,
    detect_long_trade_entries,
    summarise_short_trade_entries,
)
from smc_model_v3 import detect_v3_trade_entries, resample_to_4h

MARKET_LABEL = "US30 Proxy - Dow Futures"
TICKER = "YM=F"
LONDON_START = "08:00"
LONDON_END = "13:00"
INTRADAY_PERIODS_TO_TRY = ["60d", "30d", "15d", "7d"]

OUTPUT_PATH = "quant_lab_data.json"


def load_all_data_for_ticker(ticker):
    for period in INTRADAY_PERIODS_TO_TRY:
        data_15m = load_intraday_data(ticker=ticker, period=period, interval="15m")
        data_5m = load_intraday_data(ticker=ticker, period=period, interval="5m")

        if not data_15m.empty and not data_5m.empty:
            return data_15m, data_5m, period

    return pd.DataFrame(), pd.DataFrame(), "No data"


def load_all_data():
    return load_all_data_for_ticker(TICKER)


V3_INSTRUMENTS = {
    "US30": "YM=F",
    "USDZAR": "ZAR=X",
}


def build_v3_section(us30_data_15m, us30_data_5m):
    """
    Runs the SMC V3 model (HTF trend context, order block tagging, adjusted
    session windows, TP1/TP2 partial-close exits) across BOTH US30 and
    US500, and pools the results into one combined JSON-ready summary -
    the two instruments are tagged per-row but reported together.
    """
    tables = []

    for instrument, ticker in V3_INSTRUMENTS.items():
        if ticker == TICKER:
            data_15m, data_5m = us30_data_15m, us30_data_5m
        else:
            data_15m, data_5m, _ = load_all_data_for_ticker(ticker)

        if data_15m.empty or data_5m.empty:
            print(f"No intraday data for {instrument} ({ticker}) - skipping.", file=sys.stderr)
            continue

        data_1h = load_intraday_data(ticker=ticker, period="730d", interval="1h")
        data_4h = resample_to_4h(data_1h) if not data_1h.empty else pd.DataFrame()

        tables.append(detect_v3_trade_entries(data_15m, data_5m, data_4h, "Short", instrument))
        tables.append(detect_v3_trade_entries(data_15m, data_5m, data_4h, "Long", instrument))

    table = pd.concat([t for t in tables if not t.empty], ignore_index=True) if tables else pd.DataFrame()

    if table.empty:
        return {"available": False}

    total_days = int(table["Date"].nunique())
    entries = table[table["Entry Triggered"] == True].copy()  # noqa: E712
    entry_count = len(entries)

    win_outcomes = ["Win (TP1+TP2)", "Win (TP1 only)", "Win (TP1, runner open)", "Session Close Profit"]
    loss_outcomes = ["Loss", "Session Close Loss"]

    wins = int(entries["Outcome"].isin(win_outcomes).sum())
    losses = int(entries["Outcome"].isin(loss_outcomes).sum())
    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved > 0 else 0

    r_values = entries["R Multiple"].dropna()
    total_r = float(r_values.sum()) if not r_values.empty else 0

    aligned = entries[entries["Aligned With HTF"] == True]  # noqa: E712
    counter = entries[entries["Aligned With HTF"] == False]  # noqa: E712

    def side_stats(sub):
        r = sub["R Multiple"].dropna()
        return {
            "trades": int(len(sub)),
            "totalR": round(float(r.sum()), 2) if not r.empty else 0,
            "winRate": round(
                (sub["Outcome"].isin(win_outcomes).sum() /
                 max(sub["Outcome"].isin(win_outcomes + loss_outcomes).sum(), 1)) * 100, 2
            ),
        }

    equity_curve = []
    executed = entries[entries["R Multiple"].notna()].copy()
    if not executed.empty:
        executed["Date"] = pd.to_datetime(executed["Date"])
        executed = executed.sort_values(["Date", "Entry Time"]).reset_index(drop=True)
        cum = executed["R Multiple"].cumsum()
        peak = cum.cummax()
        for i in range(len(executed)):
            equity_curve.append({
                "tradeNumber": i + 1,
                "cumulativeR": round(float(cum.iloc[i]), 4),
                "runningPeakR": round(float(peak.iloc[i]), 4),
                "drawdownR": round(float(cum.iloc[i] - peak.iloc[i]), 4),
                "direction": executed.iloc[i]["Direction"],
                "instrument": executed.iloc[i]["Instrument"],
            })

    outcome_counts = table["Outcome"].value_counts()
    outcome_summary = [{"outcome": k, "count": int(v)} for k, v in outcome_counts.items()]

    def clean(v):
        if pd.isna(v):
            return None
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, (bool,)):
            return v
        return str(v)

    display_columns = [
        "Date", "Instrument", "Direction", "HTF Trend", "Aligned With HTF", "Order Block Found",
        "Order Block Retested", "Sweep Time", "BOS Time", "Entry Time", "Entry Level",
        "Stop Level", "Target Level", "Outcome", "R Multiple", "Plain English Result",
    ]
    recent = table.sort_values(by=["Date"], ascending=False).head(20)
    recent_rows = [{c: clean(row[c]) for c in display_columns} for _, row in recent.iterrows()]

    def instrument_stats(name):
        sub = entries[entries["Instrument"] == name]
        r = sub["R Multiple"].dropna()
        return {
            "trades": int(len(sub)),
            "totalR": round(float(r.sum()), 2) if not r.empty else 0,
            "winRate": round(
                (sub["Outcome"].isin(win_outcomes).sum() /
                 max(sub["Outcome"].isin(win_outcomes + loss_outcomes).sum(), 1)) * 100, 2
            ),
        }

    display_names = {"US30": "US30", "USDZAR": "USD/ZAR", "US500": "US500"}
    instruments_label = " and ".join(display_names.get(n, n) for n in V3_INSTRUMENTS)

    return {
        "available": True,
        "modelSettings": {
            "instruments": instruments_label,
            "londonRange": "08:00–12:30",
            "tradingWindow": "12:30–21:00 (US session)",
            "htfLookbackDays": 14,
        },
        "byInstrument": {
            name: instrument_stats(name) for name in V3_INSTRUMENTS
        },
        "snapshot": {
            "totalDays": total_days,
            "entriesTriggered": entry_count,
            "winRate": round(win_rate, 2),
            "wins": wins,
            "losses": losses,
            "totalR": round(total_r, 2),
        },
        "byHtfAlignment": {
            "aligned": side_stats(aligned),
            "counterTrend": side_stats(counter),
        },
        "equityCurve": equity_curve,
        "outcomeSummary": outcome_summary,
        "recentTrades": recent_rows,
    }


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
    long_trade_table = detect_long_trade_entries(
        data_15m=data_15m,
        data_5m=data_5m,
        london_start=LONDON_START,
        london_end=LONDON_END,
    )

    if short_trade_table.empty and long_trade_table.empty:
        print("No entry model data was available.", file=sys.stderr)
        sys.exit(1)

    trade_table = pd.concat([short_trade_table, long_trade_table], ignore_index=True)
    trade_summary = summarise_short_trade_entries(trade_table)

    total_model_days = trade_table["Date"].nunique()
    entry_triggered_count = int(trade_table["Entry Triggered"].sum())

    profitable_trades = int(trade_table["Outcome"].isin(["Win", "Session Close Profit"]).sum())
    losing_trades = int(trade_table["Outcome"].isin(["Loss", "Session Close Loss"]).sum())
    flat_trades = int(trade_table["Outcome"].isin(["Break-even", "Session Close Flat"]).sum())

    session_close_profit = int((trade_table["Outcome"] == "Session Close Profit").sum())
    session_close_loss = int((trade_table["Outcome"] == "Session Close Loss").sum())
    session_close_flat = int((trade_table["Outcome"] == "Session Close Flat").sum())

    no_entry = int((trade_table["Outcome"] == "No Entry").sum())
    no_sweep = int((trade_table["Outcome"] == "No Sweep").sum())
    ambiguous = int((trade_table["Outcome"] == "Ambiguous").sum())

    resolved_trades = profitable_trades + losing_trades + flat_trades
    model_win_rate = (profitable_trades / resolved_trades * 100) if resolved_trades > 0 else 0

    average_r = trade_table["R Multiple"].dropna().mean()
    total_r = trade_table["R Multiple"].dropna().sum()
    entry_rate = (entry_triggered_count / total_model_days * 100) if total_model_days > 0 else 0

    # Also compute win rates split by direction, since a mixed model can hide
    # one side quietly doing all the work (or all the damage).
    def direction_stats(direction):
        sub = trade_table[trade_table["Direction"] == direction]
        entries = int(sub["Entry Triggered"].sum())
        wins = int(sub["Outcome"].isin(["Win", "Session Close Profit"]).sum())
        losses = int(sub["Outcome"].isin(["Loss", "Session Close Loss"]).sum())
        flats = int(sub["Outcome"].isin(["Break-even", "Session Close Flat"]).sum())
        resolved = wins + losses + flats
        win_rate = (wins / resolved * 100) if resolved > 0 else 0
        total_r_side = sub["R Multiple"].dropna().sum()
        return {
            "entriesTriggered": entries,
            "wins": wins,
            "losses": losses,
            "flat": flats,
            "winRate": round(win_rate, 2),
            "totalR": round(float(total_r_side), 2),
        }

    executed_trades = trade_table[
        (trade_table["Entry Triggered"] == True)  # noqa: E712
        & (trade_table["R Multiple"].notna())
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
                "direction": row["Direction"],
            }
            for _, row in executed_trades.iterrows()
        ]

    outcome_summary = [
        {"outcome": row["Outcome"], "count": int(row["Count"])}
        for _, row in trade_summary.iterrows()
    ]

    display_columns = [
        "Date", "Direction", "Sweep Time", "BOS Time", "Entry Time", "London High",
        "London Low", "Entry Level", "Stop Level",
        "Target Level", "Outcome", "R Multiple", "Plain English Result",
    ]
    recent_trades_df = trade_table[display_columns].copy()
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

    model_v3 = build_v3_section(data_15m, data_5m)

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "modelV3": model_v3,
        "modelSettings": {
            "market": MARKET_LABEL,
            "londonRange": f"{LONDON_START}–{LONDON_END}",
            "dataUsed": "15m + 5m",
            "dataWindow": active_period,
        },
        "snapshot": {
            "totalModelDays": int(total_model_days),
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
        "byDirection": {
            "short": direction_stats("Short"),
            "long": direction_stats("Long"),
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

    print(
        f"Wrote {OUTPUT_PATH} ({total_model_days} model days, {entry_triggered_count} entries: "
        f"{output['byDirection']['short']['entriesTriggered']} short, "
        f"{output['byDirection']['long']['entriesTriggered']} long)."
    )


if __name__ == "__main__":
    main()
