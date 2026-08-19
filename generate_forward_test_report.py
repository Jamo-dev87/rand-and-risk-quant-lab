"""
Reads the MT5 EA's trade log (US30_LiquiditySweep_Log.csv) and builds a
summary report of the live demo forward test, in the same spirit as
export_data.py's backtest report.

Usage:
    python3 generate_forward_test_report.py /path/to/US30_LiquiditySweep_Log.csv

The CSV path is normally under the MT5 terminal's data folder, e.g.:
~/Library/Application Support/MetaTrader 5/Bottles/metatrader5/drive_c/
    Program Files/MetaTrader 5/MQL5/Files/US30_LiquiditySweep_Log.csv
"""

import csv
import json
import sys
from datetime import datetime, timezone

OUTPUT_PATH = "forward_test_report.json"


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 generate_forward_test_report.py <path-to-log.csv>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows found in the log file yet.", file=sys.stderr)
        sys.exit(1)

    trades = [r for r in rows if r["EventType"] == "Trade"]
    no_trades = [r for r in rows if r["EventType"] == "No Trade"]

    total_days = len(rows)
    entries_triggered = len(trades)

    wins = [t for t in trades if t["Outcome"].startswith("Win")]
    losses = [t for t in trades if t["Outcome"] == "Loss"]
    breakevens = [t for t in trades if t["Outcome"] == "Break-even"]

    resolved = len(wins) + len(losses) + len(breakevens)
    win_rate = (len(wins) / resolved * 100) if resolved > 0 else 0

    r_multiples = [parse_float(t["RMultiple"]) for t in trades]
    r_multiples = [r for r in r_multiples if r is not None]
    total_r = sum(r_multiples) if r_multiples else 0
    average_r = (total_r / len(r_multiples)) if r_multiples else None

    equity_curve = []
    cumulative_r = 0
    running_peak = 0
    for i, r in enumerate(r_multiples, start=1):
        cumulative_r += r
        running_peak = max(running_peak, cumulative_r)
        equity_curve.append({
            "tradeNumber": i,
            "cumulativeR": round(cumulative_r, 4),
            "runningPeakR": round(running_peak, 4),
            "drawdownR": round(cumulative_r - running_peak, 4),
        })

    max_drawdown = min([p["drawdownR"] for p in equity_curve], default=0)

    outcome_counts = {}
    for r in rows:
        outcome = r["Outcome"]
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    outcome_summary = [{"outcome": k, "count": v} for k, v in outcome_counts.items()]

    balances = [parse_float(r["BalanceAfter"]) for r in rows if parse_float(r["BalanceAfter"]) is not None]
    starting_balance = parse_float(rows[0]["BalanceBefore"])
    current_balance = balances[-1] if balances else None
    profit_amounts = [parse_float(t["ProfitAmount"]) for t in trades]
    profit_amounts = [p for p in profit_amounts if p is not None]
    total_profit = sum(profit_amounts) if profit_amounts else 0

    dates = sorted(set(r["Date"] for r in rows if r["Date"]))

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "testPeriod": {
            "firstDay": dates[0] if dates else None,
            "lastDay": dates[-1] if dates else None,
            "totalDays": total_days,
        },
        "account": {
            "startingBalance": starting_balance,
            "currentBalance": current_balance,
            "totalProfit": round(total_profit, 2),
        },
        "snapshot": {
            "totalDays": total_days,
            "entriesTriggered": entries_triggered,
            "entryRate": round(entries_triggered / total_days * 100, 2) if total_days else 0,
            "winRate": round(win_rate, 2),
            "wins": len(wins),
            "losses": len(losses),
            "breakevens": len(breakevens),
            "averageR": round(average_r, 2) if average_r is not None else None,
            "totalR": round(total_r, 2),
            "maxRDrawdown": round(max_drawdown, 2),
        },
        "equityCurve": equity_curve,
        "outcomeSummary": outcome_summary,
        "trades": trades,
        "noTradeDays": no_trades,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"  {total_days} days logged, {entries_triggered} trades taken")
    print(f"  Win rate: {win_rate:.1f}% | Total R: {total_r:.2f} | Total P/L: {total_profit:.2f}")
    print(f"  Balance: {starting_balance} -> {current_balance}")


if __name__ == "__main__":
    main()
