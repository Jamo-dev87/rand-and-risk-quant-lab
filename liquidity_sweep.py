import pandas as pd
import yfinance as yf


def load_intraday_data(ticker="YM=F", period="60d", interval="15m"):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    if "Datetime" in data.columns:
        data = data.rename(columns={"Datetime": "Timestamp"})
    elif "Date" in data.columns:
        data = data.rename(columns={"Date": "Timestamp"})

    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    if data["Timestamp"].dt.tz is None:
        data["Timestamp"] = data["Timestamp"].dt.tz_localize("UTC")

    data["Timestamp"] = data["Timestamp"].dt.tz_convert("Europe/London")

    data["Date"] = data["Timestamp"].dt.date
    data["Time"] = data["Timestamp"].dt.time

    return data


def detect_london_sweeps(data, london_start="08:00", london_end="13:00"):
    if data.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(london_start).time()
    london_end_time = pd.to_datetime(london_end).time()

    for trading_date, day_data in data.groupby("Date"):
        day_data = day_data.sort_values("Timestamp").copy()

        london_session = day_data[
            (day_data["Time"] >= london_start_time) &
            (day_data["Time"] <= london_end_time)
        ]

        post_london = day_data[day_data["Time"] > london_end_time]

        if london_session.empty or post_london.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        swept_high = post_london["High"].max() > london_high
        swept_low = post_london["Low"].min() < london_low

        if swept_high and swept_low:
            sweep_type = "Swept Both"
        elif swept_high:
            sweep_type = "Swept High"
        elif swept_low:
            sweep_type = "Swept Low"
        else:
            sweep_type = "No Sweep"

        day_close = day_data["Close"].iloc[-1]

        if day_close > london_high:
            close_location = "Closed Above London High"
        elif day_close < london_low:
            close_location = "Closed Below London Low"
        else:
            close_location = "Closed Inside London Range"

        results.append(
            {
                "Date": trading_date,
                "London High": london_high,
                "London Low": london_low,
                "Swept High": swept_high,
                "Swept Low": swept_low,
                "Sweep Type": sweep_type,
                "Day Close": day_close,
                "Close Location": close_location
            }
        )

    return pd.DataFrame(results)


def summarise_sweeps(sweep_table):
    if sweep_table.empty:
        return pd.DataFrame()

    summary = sweep_table["Sweep Type"].value_counts().reset_index()
    summary.columns = ["Sweep Type", "Count"]
    summary["Percentage"] = summary["Count"] / summary["Count"].sum() * 100

    return summary


def detect_range_fills(data, london_start="08:00", london_end="13:00"):
    if data.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(london_start).time()
    london_end_time = pd.to_datetime(london_end).time()

    for trading_date, day_data in data.groupby("Date"):
        day_data = day_data.sort_values("Timestamp").copy()

        london_session = day_data[
            (day_data["Time"] >= london_start_time) &
            (day_data["Time"] <= london_end_time)
        ]

        post_london = day_data[day_data["Time"] > london_end_time]

        if london_session.empty or post_london.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        first_sweep = "No Sweep"
        first_sweep_time = ""
        opposite_reached = False
        opposite_sweep_time = ""
        range_fill_result = "No Sweep"

        for _, row in post_london.iterrows():
            high_swept = row["High"] > london_high
            low_swept = row["Low"] < london_low

            if high_swept and low_swept:
                first_sweep = "Both in Same Candle"
                first_sweep_time = row["Timestamp"]
                opposite_reached = True
                opposite_sweep_time = row["Timestamp"]
                range_fill_result = "Filled Same Candle"
                break

            if high_swept:
                first_sweep = "High First"
                first_sweep_time = row["Timestamp"]

                later_data = post_london[post_london["Timestamp"] > first_sweep_time]
                later_low_sweep = later_data[later_data["Low"] < london_low]

                if not later_low_sweep.empty:
                    opposite_reached = True
                    opposite_sweep_time = later_low_sweep["Timestamp"].iloc[0]
                    range_fill_result = "High First → Low Reached"
                else:
                    range_fill_result = "High Only"

                break

            if low_swept:
                first_sweep = "Low First"
                first_sweep_time = row["Timestamp"]

                later_data = post_london[post_london["Timestamp"] > first_sweep_time]
                later_high_sweep = later_data[later_data["High"] > london_high]

                if not later_high_sweep.empty:
                    opposite_reached = True
                    opposite_sweep_time = later_high_sweep["Timestamp"].iloc[0]
                    range_fill_result = "Low First → High Reached"
                else:
                    range_fill_result = "Low Only"

                break

        results.append(
            {
                "Date": trading_date,
                "London High": london_high,
                "London Low": london_low,
                "First Sweep": first_sweep,
                "First Sweep Time": first_sweep_time,
                "Opposite Side Reached": opposite_reached,
                "Opposite Sweep Time": opposite_sweep_time,
                "Range Fill Result": range_fill_result
            }
        )

    range_fill_table = pd.DataFrame(results)

    if not range_fill_table.empty:
        range_fill_table["First Sweep Time"] = pd.to_datetime(
            range_fill_table["First Sweep Time"],
            errors="coerce"
        ).dt.strftime("%H:%M")

        range_fill_table["Opposite Sweep Time"] = pd.to_datetime(
            range_fill_table["Opposite Sweep Time"],
            errors="coerce"
        ).dt.strftime("%H:%M")

        range_fill_table["First Sweep Time"] = range_fill_table["First Sweep Time"].fillna("")
        range_fill_table["Opposite Sweep Time"] = range_fill_table["Opposite Sweep Time"].fillna("")

    return range_fill_table


def summarise_range_fills(range_fill_table):
    if range_fill_table.empty:
        return pd.DataFrame()

    summary = range_fill_table["Range Fill Result"].value_counts().reset_index()
    summary.columns = ["Range Fill Result", "Count"]
    summary["Percentage"] = summary["Count"] / summary["Count"].sum() * 100

    return summary


def detect_liquidity_grabs(data, london_start="08:00", london_end="13:00"):
    if data.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(london_start).time()
    london_end_time = pd.to_datetime(london_end).time()

    for trading_date, day_data in data.groupby("Date"):
        day_data = day_data.sort_values("Timestamp").copy()

        london_session = day_data[
            (day_data["Time"] >= london_start_time) &
            (day_data["Time"] <= london_end_time)
        ]

        post_london = day_data[day_data["Time"] > london_end_time]

        if london_session.empty or post_london.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        first_sweep = "No Sweep"
        sweep_time = ""
        sweep_candle_open = None
        sweep_candle_high = None
        sweep_candle_low = None
        sweep_candle_close = None
        liquidity_grab_type = "No Sweep"
        directional_idea = "No Setup"

        opposite_side_reached = False
        opposite_sweep_time = ""
        range_fill_result = "No Sweep"

        for _, row in post_london.iterrows():
            high_swept = row["High"] > london_high
            low_swept = row["Low"] < london_low

            if high_swept and low_swept:
                first_sweep = "Both in Same Candle"
                sweep_time = row["Timestamp"]
                sweep_candle_open = row["Open"]
                sweep_candle_high = row["High"]
                sweep_candle_low = row["Low"]
                sweep_candle_close = row["Close"]
                liquidity_grab_type = "Both Sides Swept"
                directional_idea = "Unclear"
                opposite_side_reached = True
                opposite_sweep_time = row["Timestamp"]
                range_fill_result = "Filled Same Candle"
                break

            if high_swept:
                first_sweep = "High First"
                sweep_time = row["Timestamp"]
                sweep_candle_open = row["Open"]
                sweep_candle_high = row["High"]
                sweep_candle_low = row["Low"]
                sweep_candle_close = row["Close"]

                if row["Close"] < london_high:
                    liquidity_grab_type = "High Wick Rejection"
                    directional_idea = "Potential Short"
                else:
                    liquidity_grab_type = "High Close Breakout"
                    directional_idea = "Breakout Risk"

                later_data = post_london[post_london["Timestamp"] > sweep_time]
                later_low_sweep = later_data[later_data["Low"] < london_low]

                if not later_low_sweep.empty:
                    opposite_side_reached = True
                    opposite_sweep_time = later_low_sweep["Timestamp"].iloc[0]
                    range_fill_result = "High First → Low Reached"
                else:
                    range_fill_result = "High Only"

                break

            if low_swept:
                first_sweep = "Low First"
                sweep_time = row["Timestamp"]
                sweep_candle_open = row["Open"]
                sweep_candle_high = row["High"]
                sweep_candle_low = row["Low"]
                sweep_candle_close = row["Close"]

                if row["Close"] > london_low:
                    liquidity_grab_type = "Low Wick Rejection"
                    directional_idea = "Potential Long"
                else:
                    liquidity_grab_type = "Low Close Breakout"
                    directional_idea = "Breakout Risk"

                later_data = post_london[post_london["Timestamp"] > sweep_time]
                later_high_sweep = later_data[later_data["High"] > london_high]

                if not later_high_sweep.empty:
                    opposite_side_reached = True
                    opposite_sweep_time = later_high_sweep["Timestamp"].iloc[0]
                    range_fill_result = "Low First → High Reached"
                else:
                    range_fill_result = "Low Only"

                break

        results.append(
            {
                "Date": trading_date,
                "London High": london_high,
                "London Low": london_low,
                "First Sweep": first_sweep,
                "Sweep Time": sweep_time,
                "Sweep Candle Open": sweep_candle_open,
                "Sweep Candle High": sweep_candle_high,
                "Sweep Candle Low": sweep_candle_low,
                "Sweep Candle Close": sweep_candle_close,
                "Liquidity Grab Type": liquidity_grab_type,
                "Directional Idea": directional_idea,
                "Opposite Side Reached": opposite_side_reached,
                "Opposite Sweep Time": opposite_sweep_time,
                "Range Fill Result": range_fill_result
            }
        )

    grab_table = pd.DataFrame(results)

    if not grab_table.empty:
        grab_table["Sweep Time"] = pd.to_datetime(
            grab_table["Sweep Time"],
            errors="coerce"
        ).dt.strftime("%H:%M")

        grab_table["Opposite Sweep Time"] = pd.to_datetime(
            grab_table["Opposite Sweep Time"],
            errors="coerce"
        ).dt.strftime("%H:%M")

        grab_table["Sweep Time"] = grab_table["Sweep Time"].fillna("")
        grab_table["Opposite Sweep Time"] = grab_table["Opposite Sweep Time"].fillna("")

    return grab_table


def summarise_liquidity_grabs(grab_table):
    if grab_table.empty:
        return pd.DataFrame()

    summary = grab_table["Liquidity Grab Type"].value_counts().reset_index()
    summary.columns = ["Liquidity Grab Type", "Count"]
    summary["Percentage"] = summary["Count"] / summary["Count"].sum() * 100

    return summary


def summarise_liquidity_grab_outcomes(grab_table):
    if grab_table.empty:
        return pd.DataFrame()

    summary = (
        grab_table
        .groupby("Liquidity Grab Type")
        .agg(
            Count=("Liquidity Grab Type", "count"),
            Range_Fills=("Opposite Side Reached", "sum")
        )
        .reset_index()
    )

    summary["Range Fill Rate"] = summary["Range_Fills"] / summary["Count"] * 100
    summary = summary.rename(columns={"Range_Fills": "Range Fills"})

    return summary


def detect_short_setup_feasibility(data, london_start="08:00", london_end="13:00"):
    if data.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(london_start).time()
    london_end_time = pd.to_datetime(london_end).time()

    for trading_date, day_data in data.groupby("Date"):
        day_data = day_data.sort_values("Timestamp").copy()

        london_session = day_data[
            (day_data["Time"] >= london_start_time) &
            (day_data["Time"] <= london_end_time)
        ]

        post_london = day_data[day_data["Time"] > london_end_time]

        if london_session.empty or post_london.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        setup_found = False

        for _, sweep_row in post_london.iterrows():
            high_swept = sweep_row["High"] > london_high
            low_swept = sweep_row["Low"] < london_low

            if low_swept and not high_swept:
                break

            if high_swept:
                setup_found = True

                sweep_time = sweep_row["Timestamp"]
                sweep_high = sweep_row["High"]
                sweep_close = sweep_row["Close"]

                stop_level = sweep_high
                target_level = london_low
                entry_proxy = sweep_close

                risk_points = stop_level - entry_proxy
                reward_points = entry_proxy - target_level

                if risk_points > 0:
                    planned_r_multiple = reward_points / risk_points
                else:
                    planned_r_multiple = None

                later_data = post_london[post_london["Timestamp"] > sweep_time]

                outcome = "Unresolved"
                outcome_time = ""
                explanation = "Neither target nor stop was reached after the sweep."

                for _, future_row in later_data.iterrows():
                    target_hit = future_row["Low"] <= target_level
                    stop_hit = future_row["High"] >= stop_level

                    if target_hit and stop_hit:
                        outcome = "Ambiguous"
                        outcome_time = future_row["Timestamp"]
                        explanation = (
                            "Target and stop were both touched inside the same 15-minute candle, "
                            "so the order cannot be confirmed from OHLC data."
                        )
                        break

                    if target_hit:
                        outcome = "Win"
                        outcome_time = future_row["Timestamp"]
                        explanation = "London low was reached before the sweep high was broken."
                        break

                    if stop_hit:
                        outcome = "Loss"
                        outcome_time = future_row["Timestamp"]
                        explanation = "Sweep high was broken before London low was reached."
                        break

                results.append(
                    {
                        "Date": trading_date,
                        "Sweep Time": sweep_time,
                        "London High": london_high,
                        "London Low": london_low,
                        "Sweep High": sweep_high,
                        "Sweep Close": sweep_close,
                        "Entry Proxy": entry_proxy,
                        "Stop Level": stop_level,
                        "Target Level": target_level,
                        "Risk Points": risk_points,
                        "Reward Points": reward_points,
                        "Planned R Multiple": planned_r_multiple,
                        "Outcome": outcome,
                        "Outcome Time": outcome_time,
                        "Plain English Result": explanation
                    }
                )

                break

        if not setup_found:
            continue

    feasibility_table = pd.DataFrame(results)

    if not feasibility_table.empty:
        feasibility_table["Sweep Time"] = pd.to_datetime(
            feasibility_table["Sweep Time"],
            errors="coerce"
        ).dt.strftime("%H:%M")

        feasibility_table["Outcome Time"] = pd.to_datetime(
            feasibility_table["Outcome Time"],
            errors="coerce"
        ).dt.strftime("%H:%M")

        feasibility_table["Sweep Time"] = feasibility_table["Sweep Time"].fillna("")
        feasibility_table["Outcome Time"] = feasibility_table["Outcome Time"].fillna("")

    return feasibility_table


def summarise_short_setup_feasibility(feasibility_table):
    if feasibility_table.empty:
        return pd.DataFrame()

    summary = feasibility_table["Outcome"].value_counts().reset_index()
    summary.columns = ["Outcome", "Count"]
    summary["Percentage"] = summary["Count"] / summary["Count"].sum() * 100

    return summary


# ============================================================
# SHORT ENTRY MODEL V2
# ============================================================

def is_bearish_candle(row):
    return row["Close"] < row["Open"]


def is_bullish_candle(row):
    return row["Close"] > row["Open"]


def detect_bearish_bos_v2(data_after_sweep):
    """
    Strict bearish BOS model.

    Logic:
    - Looks for a meaningful bearish structure level created by 3 consecutive bearish candles.
    - The BOS level is the lowest low of that 3-candle bearish sequence.
    - A valid BOS requires a later 5-minute candle to CLOSE below that level.
    - A wick below the level does not count.
    """

    if data_after_sweep.empty or len(data_after_sweep) < 8:
        return None

    data_after_sweep = data_after_sweep.sort_values("Timestamp").reset_index(drop=True)

    current_structure_low = None
    current_structure_time = None
    current_structure_start_time = None

    for i in range(2, len(data_after_sweep)):
        row = data_after_sweep.iloc[i]

        candle_1 = data_after_sweep.iloc[i - 2]
        candle_2 = data_after_sweep.iloc[i - 1]
        candle_3 = data_after_sweep.iloc[i]

        three_bearish_candles = (
            is_bearish_candle(candle_1) and
            is_bearish_candle(candle_2) and
            is_bearish_candle(candle_3)
        )

        if three_bearish_candles:
            current_structure_low = min(
                candle_1["Low"],
                candle_2["Low"],
                candle_3["Low"]
            )

            current_structure_start_time = candle_1["Timestamp"]
            current_structure_time = candle_3["Timestamp"]

            continue

        if current_structure_low is not None:
            close_breaks_structure = row["Close"] < current_structure_low
            wick_only_break = (
                row["Low"] < current_structure_low and
                row["Close"] >= current_structure_low
            )

            if close_breaks_structure:
                return {
                    "BOS Time": row["Timestamp"],
                    "BOS Close": row["Close"],
                    "BOS Low": row["Low"],
                    "Broken Structure Low": current_structure_low,
                    "Structure Start Time": current_structure_start_time,
                    "Structure End Time": current_structure_time,
                    "BOS Type": "Close Below Structure"
                }

            if wick_only_break:
                continue

    return None


def find_short_retest_entry_v2(data_after_bos, london_high, liquidity_high):
    """
    Short entry model v2.

    Logic:
    - After BOS, price must retrace back into the original liquidity area.
    - Liquidity area = London High to original liquidity high.
    - Setup is invalid if price breaks above the original liquidity high before entry.
    - Once price retests the liquidity area, wait for 3 consecutive bullish candles.
    - Entry is at the close of the 3rd bullish candle.
    """

    if data_after_bos.empty:
        return {
            "Entry Triggered": False,
            "Entry Time": "",
            "Entry Level": None,
            "Retest Time": "",
            "Retest High": None,
            "Entry Failure Reason": "No 5-minute data after BOS."
        }

    data_after_bos = data_after_bos.sort_values("Timestamp").reset_index(drop=True)

    retest_seen = False
    retest_time = ""
    retest_high = None
    bullish_count = 0

    for _, row in data_after_bos.iterrows():
        if row["High"] > liquidity_high:
            return {
                "Entry Triggered": False,
                "Entry Time": "",
                "Entry Level": None,
                "Retest Time": retest_time,
                "Retest High": retest_high,
                "Entry Failure Reason": (
                    "Setup invalidated because price broke above the original liquidity high before entry."
                )
            }

        candle_retests_liquidity_zone = (
            row["High"] >= london_high and
            row["High"] <= liquidity_high
        )

        if candle_retests_liquidity_zone and not retest_seen:
            retest_seen = True
            retest_time = row["Timestamp"]
            retest_high = row["High"]

        if retest_seen:
            if is_bullish_candle(row):
                bullish_count += 1
            else:
                bullish_count = 0

            if bullish_count >= 3:
                return {
                    "Entry Triggered": True,
                    "Entry Time": row["Timestamp"],
                    "Entry Level": row["Close"],
                    "Retest Time": retest_time,
                    "Retest High": retest_high,
                    "Entry Failure Reason": ""
                }

    if not retest_seen:
        failure_reason = "Price did not retest the original liquidity zone after BOS."
    else:
        failure_reason = (
            "Price retested the liquidity zone but did not print 3 consecutive bullish candles "
            "before the data ended."
        )

    return {
        "Entry Triggered": False,
        "Entry Time": "",
        "Entry Level": None,
        "Retest Time": retest_time,
        "Retest High": retest_high,
        "Entry Failure Reason": failure_reason
    }


def simulate_short_trade_v2(trade_data, entry_level, stop_level, target_level):
    """
    Simulates the short trade after entry.

    Rules:
    - Stop = original liquidity high.
    - Target = London low.
    - Break-even trigger = 2R.
    - Once 2R is reached, stop moves to entry.
    - If no stop/target/BE resolution occurs, close at the final available session close.
    """

    if trade_data.empty:
        return {
            "Outcome": "Unresolved",
            "Outcome Time": "",
            "R Multiple": None,
            "BE Trigger": None,
            "Plain English Result": "Entry triggered, but there was no data after entry."
        }

    risk_points = stop_level - entry_level
    reward_points = entry_level - target_level

    if risk_points <= 0:
        return {
            "Outcome": "Invalid Risk",
            "Outcome Time": "",
            "R Multiple": None,
            "BE Trigger": None,
            "Plain English Result": "Invalid risk: stop level is not above entry level for the short trade."
        }

    if reward_points <= 0:
        return {
            "Outcome": "Invalid Reward",
            "Outcome Time": "",
            "R Multiple": None,
            "BE Trigger": None,
            "Plain English Result": "Invalid reward: target is not below entry level for the short trade."
        }

    planned_r_multiple = reward_points / risk_points
    be_trigger = entry_level - (2 * risk_points)
    moved_to_be = False

    for _, row in trade_data.iterrows():
        stop_hit = row["High"] >= stop_level
        target_hit = row["Low"] <= target_level

        if not moved_to_be:
            be_hit = row["Low"] <= be_trigger
        else:
            be_hit = False

        if stop_hit and target_hit:
            return {
                "Outcome": "Ambiguous",
                "Outcome Time": row["Timestamp"],
                "R Multiple": None,
                "BE Trigger": be_trigger,
                "Plain English Result": (
                    "Stop and target were both touched inside the same 5-minute candle, "
                    "so order cannot be confirmed from OHLC data."
                )
            }

        if moved_to_be:
            be_stop_hit = row["High"] >= entry_level

            if be_stop_hit and target_hit:
                return {
                    "Outcome": "Ambiguous",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": None,
                    "BE Trigger": be_trigger,
                    "Plain English Result": (
                        "Break-even stop and target were both touched inside the same 5-minute candle."
                    )
                }

            if target_hit:
                return {
                    "Outcome": "Win",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": planned_r_multiple,
                    "BE Trigger": be_trigger,
                    "Plain English Result": (
                        "Target was reached after price had moved far enough to trigger break-even."
                    )
                }

            if be_stop_hit:
                return {
                    "Outcome": "Break-even",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": 0,
                    "BE Trigger": be_trigger,
                    "Plain English Result": (
                        "Price reached 2R, stop moved to break-even, then break-even was hit."
                    )
                }

        else:
            if stop_hit:
                return {
                    "Outcome": "Loss",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": -1,
                    "BE Trigger": be_trigger,
                    "Plain English Result": "Stop-loss was hit before target or break-even trigger."
                }

            if target_hit:
                return {
                    "Outcome": "Win",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": planned_r_multiple,
                    "BE Trigger": be_trigger,
                    "Plain English Result": "London low target was reached before stop-loss."
                }

            if be_hit:
                moved_to_be = True

    final_row = trade_data.iloc[-1]
    final_close = final_row["Close"]
    final_time = final_row["Timestamp"]

    final_r = (entry_level - final_close) / risk_points

    if abs(final_r) < 1e-9:
        outcome = "Session Close Flat"
        explanation = (
            "Trade did not hit stop, target, or break-even resolution, "
            "so it was closed flat at the final available session price."
        )
        final_r = 0
    elif final_r > 0:
        outcome = "Session Close Profit"
        explanation = (
            "Trade did not hit stop, target, or break-even resolution, "
            "so it was closed in profit at the final available session price."
        )
    else:
        outcome = "Session Close Loss"
        explanation = (
            "Trade did not hit stop, target, or break-even resolution, "
            "so it was closed at a loss at the final available session price."
        )

    return {
        "Outcome": outcome,
        "Outcome Time": final_time,
        "R Multiple": final_r,
        "BE Trigger": be_trigger,
        "Plain English Result": explanation
    }


def detect_short_trade_entries(data_15m, data_5m, london_start="08:00", london_end="13:00"):
    """
    Short Entry Model v2.

    Updated strategy logic:
    1. London range is defined from london_start to london_end.
    2. After London ends, price must trade above the London high.
       - Wick above is valid.
       - Close above is also valid.
    3. The original liquidity high is recorded.
    4. On 5-minute data, wait for a strict bearish BOS:
       - BOS level comes from the nearest 3-candle bearish structure.
       - BOS must CLOSE below the structure low.
       - Wick below does not count.
    5. After BOS, price must retest the original liquidity zone.
       - Zone = London High to liquidity high.
       - If price breaks above liquidity high before entry, setup is invalid.
    6. After retest, wait for 3 consecutive bullish candles.
    7. Enter short at the close of the 3rd bullish candle.
    8. Stop = original liquidity high.
    9. Target = London low.
    10. Move stop to break-even after 2R.
    11. If unresolved, close at final available session close and calculate R.
    """

    if data_15m.empty or data_5m.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(london_start).time()
    london_end_time = pd.to_datetime(london_end).time()

    for trading_date, day_data_15m in data_15m.groupby("Date"):
        day_data_15m = day_data_15m.sort_values("Timestamp").copy()
        day_data_5m = data_5m[data_5m["Date"] == trading_date].sort_values("Timestamp").copy()

        if day_data_5m.empty:
            continue

        london_session = day_data_15m[
            (day_data_15m["Time"] >= london_start_time) &
            (day_data_15m["Time"] <= london_end_time)
        ]

        post_london_15m = day_data_15m[day_data_15m["Time"] > london_end_time]

        if london_session.empty or post_london_15m.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        high_sweep_rows = post_london_15m[post_london_15m["High"] > london_high]

        if high_sweep_rows.empty:
            results.append(
                {
                    "Date": trading_date,
                    "Direction": "Short",
                    "London High": london_high,
                    "London Low": london_low,
                    "Sweep Time": "",
                    "Sweep High": None,
                    "Sweep Close": None,
                    "BOS Time": "",
                    "Broken Swing Low": None,
                    "BOS Low": None,
                    "Entry Level": None,
                    "Stop Level": None,
                    "Target Level": london_low,
                    "Risk Points": None,
                    "Reward Points": None,
                    "Planned R Multiple": None,
                    "BE Trigger": None,
                    "Entry Triggered": False,
                    "Entry Time": "",
                    "Outcome": "No Sweep",
                    "Outcome Time": "",
                    "R Multiple": None,
                    "Plain English Result": (
                        "No London high liquidity event occurred after the London range closed."
                    )
                }
            )
            continue

        sweep_row = high_sweep_rows.iloc[0]
        sweep_time = sweep_row["Timestamp"]
        sweep_high = sweep_row["High"]
        sweep_close = sweep_row["Close"]

        data_after_sweep_5m = day_data_5m[day_data_5m["Timestamp"] > sweep_time].copy()

        bos = detect_bearish_bos_v2(data_after_sweep_5m)

        if bos is None:
            results.append(
                {
                    "Date": trading_date,
                    "Direction": "Short",
                    "London High": london_high,
                    "London Low": london_low,
                    "Sweep Time": sweep_time,
                    "Sweep High": sweep_high,
                    "Sweep Close": sweep_close,
                    "BOS Time": "",
                    "Broken Swing Low": None,
                    "BOS Low": None,
                    "Entry Level": None,
                    "Stop Level": sweep_high,
                    "Target Level": london_low,
                    "Risk Points": None,
                    "Reward Points": None,
                    "Planned R Multiple": None,
                    "BE Trigger": None,
                    "Entry Triggered": False,
                    "Entry Time": "",
                    "Outcome": "No BOS",
                    "Outcome Time": "",
                    "R Multiple": None,
                    "Plain English Result": (
                        "London high was swept, but no strict bearish BOS was detected. "
                        "BOS requires a candle close below a 3-bearish-candle structure low."
                    )
                }
            )
            continue

        bos_time = bos["BOS Time"]

        pre_bos_5m = day_data_5m[
            (day_data_5m["Timestamp"] >= sweep_time) &
            (day_data_5m["Timestamp"] <= bos_time)
        ].copy()

        if pre_bos_5m.empty:
            liquidity_high = sweep_high
        else:
            liquidity_high = max(sweep_high, pre_bos_5m["High"].max())

        data_after_bos_5m = day_data_5m[day_data_5m["Timestamp"] > bos_time].copy()

        entry = find_short_retest_entry_v2(
            data_after_bos=data_after_bos_5m,
            london_high=london_high,
            liquidity_high=liquidity_high
        )

        if not entry["Entry Triggered"]:
            results.append(
                {
                    "Date": trading_date,
                    "Direction": "Short",
                    "London High": london_high,
                    "London Low": london_low,
                    "Sweep Time": sweep_time,
                    "Sweep High": liquidity_high,
                    "Sweep Close": sweep_close,
                    "BOS Time": bos_time,
                    "Broken Swing Low": bos["Broken Structure Low"],
                    "BOS Low": bos["BOS Low"],
                    "Entry Level": None,
                    "Stop Level": liquidity_high,
                    "Target Level": london_low,
                    "Risk Points": None,
                    "Reward Points": None,
                    "Planned R Multiple": None,
                    "BE Trigger": None,
                    "Entry Triggered": False,
                    "Entry Time": "",
                    "Outcome": "No Entry",
                    "Outcome Time": "",
                    "R Multiple": None,
                    "Plain English Result": entry["Entry Failure Reason"]
                }
            )
            continue

        entry_level = entry["Entry Level"]
        entry_time = entry["Entry Time"]
        stop_level = liquidity_high
        target_level = london_low

        risk_points = stop_level - entry_level
        reward_points = entry_level - target_level

        if risk_points > 0 and reward_points > 0:
            planned_r_multiple = reward_points / risk_points
        else:
            planned_r_multiple = None

        trade_data = day_data_5m[day_data_5m["Timestamp"] > entry_time].copy()

        outcome = simulate_short_trade_v2(
            trade_data=trade_data,
            entry_level=entry_level,
            stop_level=stop_level,
            target_level=target_level
        )

        results.append(
            {
                "Date": trading_date,
                "Direction": "Short",
                "London High": london_high,
                "London Low": london_low,
                "Sweep Time": sweep_time,
                "Sweep High": liquidity_high,
                "Sweep Close": sweep_close,
                "BOS Time": bos_time,
                "Broken Swing Low": bos["Broken Structure Low"],
                "BOS Low": bos["BOS Low"],
                "Entry Level": entry_level,
                "Stop Level": stop_level,
                "Target Level": target_level,
                "Risk Points": risk_points,
                "Reward Points": reward_points,
                "Planned R Multiple": planned_r_multiple,
                "BE Trigger": outcome["BE Trigger"],
                "Entry Triggered": True,
                "Entry Time": entry_time,
                "Outcome": outcome["Outcome"],
                "Outcome Time": outcome["Outcome Time"],
                "R Multiple": outcome["R Multiple"],
                "Plain English Result": outcome["Plain English Result"]
            }
        )

    trade_table = pd.DataFrame(results)

    if not trade_table.empty:
        for col in ["Sweep Time", "BOS Time", "Entry Time", "Outcome Time"]:
            trade_table[col] = pd.to_datetime(
                trade_table[col],
                errors="coerce"
            ).dt.strftime("%H:%M")

            trade_table[col] = trade_table[col].fillna("")

    return trade_table


def summarise_short_trade_entries(trade_table):
    if trade_table.empty:
        return pd.DataFrame()

    summary = trade_table["Outcome"].value_counts().reset_index()
    summary.columns = ["Outcome", "Count"]
    summary["Percentage"] = summary["Count"] / summary["Count"].sum() * 100

    return summary


# ============================================================
# LONG ENTRY MODEL V2 (mirror of the short model)
# ============================================================

def detect_bullish_bos_v2(data_after_sweep):
    """
    Strict bullish BOS model. Mirror of detect_bearish_bos_v2.

    Logic:
    - Looks for a meaningful bullish structure level created by 3 consecutive bullish candles.
    - The BOS level is the highest high of that 3-candle bullish sequence.
    - A valid BOS requires a later 5-minute candle to CLOSE above that level.
    - A wick above the level does not count.
    """

    if data_after_sweep.empty or len(data_after_sweep) < 8:
        return None

    data_after_sweep = data_after_sweep.sort_values("Timestamp").reset_index(drop=True)

    current_structure_high = None
    current_structure_time = None
    current_structure_start_time = None

    for i in range(2, len(data_after_sweep)):
        row = data_after_sweep.iloc[i]

        candle_1 = data_after_sweep.iloc[i - 2]
        candle_2 = data_after_sweep.iloc[i - 1]
        candle_3 = data_after_sweep.iloc[i]

        three_bullish_candles = (
            is_bullish_candle(candle_1) and
            is_bullish_candle(candle_2) and
            is_bullish_candle(candle_3)
        )

        if three_bullish_candles:
            current_structure_high = max(
                candle_1["High"],
                candle_2["High"],
                candle_3["High"]
            )

            current_structure_start_time = candle_1["Timestamp"]
            current_structure_time = candle_3["Timestamp"]

            continue

        if current_structure_high is not None:
            close_breaks_structure = row["Close"] > current_structure_high
            wick_only_break = (
                row["High"] > current_structure_high and
                row["Close"] <= current_structure_high
            )

            if close_breaks_structure:
                return {
                    "BOS Time": row["Timestamp"],
                    "BOS Close": row["Close"],
                    "BOS High": row["High"],
                    "Broken Structure High": current_structure_high,
                    "Structure Start Time": current_structure_start_time,
                    "Structure End Time": current_structure_time,
                    "BOS Type": "Close Above Structure"
                }

            if wick_only_break:
                continue

    return None


def find_long_retest_entry_v2(data_after_bos, london_low, liquidity_low):
    """
    Long entry model v2. Mirror of find_short_retest_entry_v2.

    Logic:
    - After BOS, price must retrace back into the original liquidity area.
    - Liquidity area = original liquidity low to London Low.
    - Setup is invalid if price breaks below the original liquidity low before entry.
    - Once price retests the liquidity area, wait for 3 consecutive bearish candles.
    - Entry is at the close of the 3rd bearish candle.
    """

    if data_after_bos.empty:
        return {
            "Entry Triggered": False,
            "Entry Time": "",
            "Entry Level": None,
            "Retest Time": "",
            "Retest Low": None,
            "Entry Failure Reason": "No 5-minute data after BOS."
        }

    data_after_bos = data_after_bos.sort_values("Timestamp").reset_index(drop=True)

    retest_seen = False
    retest_time = ""
    retest_low = None
    bearish_count = 0

    for _, row in data_after_bos.iterrows():
        if row["Low"] < liquidity_low:
            return {
                "Entry Triggered": False,
                "Entry Time": "",
                "Entry Level": None,
                "Retest Time": retest_time,
                "Retest Low": retest_low,
                "Entry Failure Reason": (
                    "Setup invalidated because price broke below the original liquidity low before entry."
                )
            }

        candle_retests_liquidity_zone = (
            row["Low"] <= london_low and
            row["Low"] >= liquidity_low
        )

        if candle_retests_liquidity_zone and not retest_seen:
            retest_seen = True
            retest_time = row["Timestamp"]
            retest_low = row["Low"]

        if retest_seen:
            if is_bearish_candle(row):
                bearish_count += 1
            else:
                bearish_count = 0

            if bearish_count >= 3:
                return {
                    "Entry Triggered": True,
                    "Entry Time": row["Timestamp"],
                    "Entry Level": row["Close"],
                    "Retest Time": retest_time,
                    "Retest Low": retest_low,
                    "Entry Failure Reason": ""
                }

    if not retest_seen:
        failure_reason = "Price did not retest the original liquidity zone after BOS."
    else:
        failure_reason = (
            "Price retested the liquidity zone but did not print 3 consecutive bearish candles "
            "before the data ended."
        )

    return {
        "Entry Triggered": False,
        "Entry Time": "",
        "Entry Level": None,
        "Retest Time": retest_time,
        "Retest Low": retest_low,
        "Entry Failure Reason": failure_reason
    }


def simulate_long_trade_v2(trade_data, entry_level, stop_level, target_level):
    """
    Simulates the long trade after entry. Mirror of simulate_short_trade_v2.

    Rules:
    - Stop = original liquidity low.
    - Target = London high.
    - Break-even trigger = 2R.
    - Once 2R is reached, stop moves to entry.
    - If no stop/target/BE resolution occurs, close at the final available session close.
    """

    if trade_data.empty:
        return {
            "Outcome": "Unresolved",
            "Outcome Time": "",
            "R Multiple": None,
            "BE Trigger": None,
            "Plain English Result": "Entry triggered, but there was no data after entry."
        }

    risk_points = entry_level - stop_level
    reward_points = target_level - entry_level

    if risk_points <= 0:
        return {
            "Outcome": "Invalid Risk",
            "Outcome Time": "",
            "R Multiple": None,
            "BE Trigger": None,
            "Plain English Result": "Invalid risk: stop level is not below entry level for the long trade."
        }

    if reward_points <= 0:
        return {
            "Outcome": "Invalid Reward",
            "Outcome Time": "",
            "R Multiple": None,
            "BE Trigger": None,
            "Plain English Result": "Invalid reward: target is not above entry level for the long trade."
        }

    planned_r_multiple = reward_points / risk_points
    be_trigger = entry_level + (2 * risk_points)
    moved_to_be = False

    for _, row in trade_data.iterrows():
        stop_hit = row["Low"] <= stop_level
        target_hit = row["High"] >= target_level

        if not moved_to_be:
            be_hit = row["High"] >= be_trigger
        else:
            be_hit = False

        if stop_hit and target_hit:
            return {
                "Outcome": "Ambiguous",
                "Outcome Time": row["Timestamp"],
                "R Multiple": None,
                "BE Trigger": be_trigger,
                "Plain English Result": (
                    "Stop and target were both touched inside the same 5-minute candle, "
                    "so order cannot be confirmed from OHLC data."
                )
            }

        if moved_to_be:
            be_stop_hit = row["Low"] <= entry_level

            if be_stop_hit and target_hit:
                return {
                    "Outcome": "Ambiguous",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": None,
                    "BE Trigger": be_trigger,
                    "Plain English Result": (
                        "Break-even stop and target were both touched inside the same 5-minute candle."
                    )
                }

            if target_hit:
                return {
                    "Outcome": "Win",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": planned_r_multiple,
                    "BE Trigger": be_trigger,
                    "Plain English Result": (
                        "Target was reached after price had moved far enough to trigger break-even."
                    )
                }

            if be_stop_hit:
                return {
                    "Outcome": "Break-even",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": 0,
                    "BE Trigger": be_trigger,
                    "Plain English Result": (
                        "Price reached 2R, stop moved to break-even, then break-even was hit."
                    )
                }

        else:
            if stop_hit:
                return {
                    "Outcome": "Loss",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": -1,
                    "BE Trigger": be_trigger,
                    "Plain English Result": "Stop-loss was hit before target or break-even trigger."
                }

            if target_hit:
                return {
                    "Outcome": "Win",
                    "Outcome Time": row["Timestamp"],
                    "R Multiple": planned_r_multiple,
                    "BE Trigger": be_trigger,
                    "Plain English Result": "London high target was reached before stop-loss."
                }

            if be_hit:
                moved_to_be = True

    final_row = trade_data.iloc[-1]
    final_close = final_row["Close"]
    final_time = final_row["Timestamp"]

    final_r = (final_close - entry_level) / risk_points

    if abs(final_r) < 1e-9:
        outcome = "Session Close Flat"
        explanation = (
            "Trade did not hit stop, target, or break-even resolution, "
            "so it was closed flat at the final available session price."
        )
        final_r = 0
    elif final_r > 0:
        outcome = "Session Close Profit"
        explanation = (
            "Trade did not hit stop, target, or break-even resolution, "
            "so it was closed in profit at the final available session price."
        )
    else:
        outcome = "Session Close Loss"
        explanation = (
            "Trade did not hit stop, target, or break-even resolution, "
            "so it was closed at a loss at the final available session price."
        )

    return {
        "Outcome": outcome,
        "Outcome Time": final_time,
        "R Multiple": final_r,
        "BE Trigger": be_trigger,
        "Plain English Result": explanation
    }


def detect_long_trade_entries(data_15m, data_5m, london_start="08:00", london_end="13:00"):
    """
    Long Entry Model v2. Mirror of detect_short_trade_entries.

    Strategy logic:
    1. London range is defined from london_start to london_end.
    2. After London ends, price must trade below the London low.
       - Wick below is valid.
       - Close below is also valid.
    3. The original liquidity low is recorded.
    4. On 5-minute data, wait for a strict bullish BOS:
       - BOS level comes from the nearest 3-candle bullish structure.
       - BOS must CLOSE above the structure high.
       - Wick above does not count.
    5. After BOS, price must retest the original liquidity zone.
       - Zone = liquidity low to London Low.
       - If price breaks below liquidity low before entry, setup is invalid.
    6. After retest, wait for 3 consecutive bearish candles.
    7. Enter long at the close of the 3rd bearish candle.
    8. Stop = original liquidity low.
    9. Target = London high.
    10. Move stop to break-even after 2R.
    11. If unresolved, close at final available session close and calculate R.
    """

    if data_15m.empty or data_5m.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(london_start).time()
    london_end_time = pd.to_datetime(london_end).time()

    for trading_date, day_data_15m in data_15m.groupby("Date"):
        day_data_15m = day_data_15m.sort_values("Timestamp").copy()
        day_data_5m = data_5m[data_5m["Date"] == trading_date].sort_values("Timestamp").copy()

        if day_data_5m.empty:
            continue

        london_session = day_data_15m[
            (day_data_15m["Time"] >= london_start_time) &
            (day_data_15m["Time"] <= london_end_time)
        ]

        post_london_15m = day_data_15m[day_data_15m["Time"] > london_end_time]

        if london_session.empty or post_london_15m.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        low_sweep_rows = post_london_15m[post_london_15m["Low"] < london_low]

        if low_sweep_rows.empty:
            results.append(
                {
                    "Date": trading_date,
                    "Direction": "Long",
                    "London High": london_high,
                    "London Low": london_low,
                    "Sweep Time": "",
                    "Sweep Low": None,
                    "Sweep Close": None,
                    "BOS Time": "",
                    "Broken Swing High": None,
                    "BOS High": None,
                    "Entry Level": None,
                    "Stop Level": None,
                    "Target Level": london_high,
                    "Risk Points": None,
                    "Reward Points": None,
                    "Planned R Multiple": None,
                    "BE Trigger": None,
                    "Entry Triggered": False,
                    "Entry Time": "",
                    "Outcome": "No Sweep",
                    "Outcome Time": "",
                    "R Multiple": None,
                    "Plain English Result": (
                        "No London low liquidity event occurred after the London range closed."
                    )
                }
            )
            continue

        sweep_row = low_sweep_rows.iloc[0]
        sweep_time = sweep_row["Timestamp"]
        sweep_low = sweep_row["Low"]
        sweep_close = sweep_row["Close"]

        data_after_sweep_5m = day_data_5m[day_data_5m["Timestamp"] > sweep_time].copy()

        bos = detect_bullish_bos_v2(data_after_sweep_5m)

        if bos is None:
            results.append(
                {
                    "Date": trading_date,
                    "Direction": "Long",
                    "London High": london_high,
                    "London Low": london_low,
                    "Sweep Time": sweep_time,
                    "Sweep Low": sweep_low,
                    "Sweep Close": sweep_close,
                    "BOS Time": "",
                    "Broken Swing High": None,
                    "BOS High": None,
                    "Entry Level": None,
                    "Stop Level": sweep_low,
                    "Target Level": london_high,
                    "Risk Points": None,
                    "Reward Points": None,
                    "Planned R Multiple": None,
                    "BE Trigger": None,
                    "Entry Triggered": False,
                    "Entry Time": "",
                    "Outcome": "No BOS",
                    "Outcome Time": "",
                    "R Multiple": None,
                    "Plain English Result": (
                        "London low was swept, but no strict bullish BOS was detected. "
                        "BOS requires a candle close above a 3-bullish-candle structure high."
                    )
                }
            )
            continue

        bos_time = bos["BOS Time"]

        pre_bos_5m = day_data_5m[
            (day_data_5m["Timestamp"] >= sweep_time) &
            (day_data_5m["Timestamp"] <= bos_time)
        ].copy()

        if pre_bos_5m.empty:
            liquidity_low = sweep_low
        else:
            liquidity_low = min(sweep_low, pre_bos_5m["Low"].min())

        data_after_bos_5m = day_data_5m[day_data_5m["Timestamp"] > bos_time].copy()

        entry = find_long_retest_entry_v2(
            data_after_bos=data_after_bos_5m,
            london_low=london_low,
            liquidity_low=liquidity_low
        )

        if not entry["Entry Triggered"]:
            results.append(
                {
                    "Date": trading_date,
                    "Direction": "Long",
                    "London High": london_high,
                    "London Low": london_low,
                    "Sweep Time": sweep_time,
                    "Sweep Low": liquidity_low,
                    "Sweep Close": sweep_close,
                    "BOS Time": bos_time,
                    "Broken Swing High": bos["Broken Structure High"],
                    "BOS High": bos["BOS High"],
                    "Entry Level": None,
                    "Stop Level": liquidity_low,
                    "Target Level": london_high,
                    "Risk Points": None,
                    "Reward Points": None,
                    "Planned R Multiple": None,
                    "BE Trigger": None,
                    "Entry Triggered": False,
                    "Entry Time": "",
                    "Outcome": "No Entry",
                    "Outcome Time": "",
                    "R Multiple": None,
                    "Plain English Result": entry["Entry Failure Reason"]
                }
            )
            continue

        entry_level = entry["Entry Level"]
        entry_time = entry["Entry Time"]
        stop_level = liquidity_low
        target_level = london_high

        risk_points = entry_level - stop_level
        reward_points = target_level - entry_level

        if risk_points > 0 and reward_points > 0:
            planned_r_multiple = reward_points / risk_points
        else:
            planned_r_multiple = None

        trade_data = day_data_5m[day_data_5m["Timestamp"] > entry_time].copy()

        outcome = simulate_long_trade_v2(
            trade_data=trade_data,
            entry_level=entry_level,
            stop_level=stop_level,
            target_level=target_level
        )

        results.append(
            {
                "Date": trading_date,
                "Direction": "Long",
                "London High": london_high,
                "London Low": london_low,
                "Sweep Time": sweep_time,
                "Sweep Low": liquidity_low,
                "Sweep Close": sweep_close,
                "BOS Time": bos_time,
                "Broken Swing High": bos["Broken Structure High"],
                "BOS High": bos["BOS High"],
                "Entry Level": entry_level,
                "Stop Level": stop_level,
                "Target Level": target_level,
                "Risk Points": risk_points,
                "Reward Points": reward_points,
                "Planned R Multiple": planned_r_multiple,
                "BE Trigger": outcome["BE Trigger"],
                "Entry Triggered": True,
                "Entry Time": entry_time,
                "Outcome": outcome["Outcome"],
                "Outcome Time": outcome["Outcome Time"],
                "R Multiple": outcome["R Multiple"],
                "Plain English Result": outcome["Plain English Result"]
            }
        )

    trade_table = pd.DataFrame(results)

    if not trade_table.empty:
        for col in ["Sweep Time", "BOS Time", "Entry Time", "Outcome Time"]:
            trade_table[col] = pd.to_datetime(
                trade_table[col],
                errors="coerce"
            ).dt.strftime("%H:%M")

            trade_table[col] = trade_table[col].fillna("")

    return trade_table


def summarise_long_trade_entries(trade_table):
    if trade_table.empty:
        return pd.DataFrame()

    summary = trade_table["Outcome"].value_counts().reset_index()
    summary.columns = ["Outcome", "Count"]
    summary["Percentage"] = summary["Count"] / summary["Count"].sum() * 100

    return summary