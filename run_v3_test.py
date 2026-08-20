import pandas as pd

from liquidity_sweep import load_intraday_data
from smc_model_v3 import detect_v3_trade_entries, resample_to_4h

TICKER = "YM=F"
PERIODS_TO_TRY = ["60d", "30d", "15d", "7d"]


def load_all():
    for period in PERIODS_TO_TRY:
        d15 = load_intraday_data(ticker=TICKER, period=period, interval="15m")
        d5 = load_intraday_data(ticker=TICKER, period=period, interval="5m")
        if not d15.empty and not d5.empty:
            return d15, d5, period
    return pd.DataFrame(), pd.DataFrame(), "No data"


def main():
    data_15m, data_5m, period = load_all()
    print(f"15m/5m data window: {period}, {len(data_15m)} 15m bars, {len(data_5m)} 5m bars")

    data_1h = load_intraday_data(ticker=TICKER, period="730d", interval="1h")
    print(f"1h bars for HTF context: {len(data_1h)}")

    data_4h = resample_to_4h(data_1h)
    print(f"4h bars after resample: {len(data_4h)}")

    short_table = detect_v3_trade_entries(data_15m, data_5m, data_4h, "Short")
    long_table = detect_v3_trade_entries(data_15m, data_5m, data_4h, "Long")

    combined = pd.concat([short_table, long_table], ignore_index=True)

    print(f"\nTotal rows: {len(combined)}")
    print(f"Entries triggered: {int(combined['Entry Triggered'].sum())}")
    print("\nOutcome counts:")
    print(combined["Outcome"].value_counts())

    print("\nHTF Trend distribution:")
    print(combined["HTF Trend"].value_counts())

    print("\nAligned with HTF trend (among triggered entries):")
    triggered = combined[combined["Entry Triggered"] == True]  # noqa: E712
    if not triggered.empty:
        print(triggered["Aligned With HTF"].value_counts())
        print(f"\nTotal R (all triggered): {triggered['R Multiple'].dropna().sum():.2f}")
        aligned = triggered[triggered["Aligned With HTF"] == True]  # noqa: E712
        not_aligned = triggered[triggered["Aligned With HTF"] == False]  # noqa: E712
        print(f"Total R (aligned with HTF): {aligned['R Multiple'].dropna().sum():.2f} over {len(aligned)} trades")
        print(f"Total R (counter to HTF): {not_aligned['R Multiple'].dropna().sum():.2f} over {len(not_aligned)} trades")

    combined.to_csv("v3_test_output.csv", index=False)
    print("\nWrote v3_test_output.csv")


if __name__ == "__main__":
    main()
