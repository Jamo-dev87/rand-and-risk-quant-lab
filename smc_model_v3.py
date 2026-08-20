"""
SMC Model V3 - adds higher-timeframe trend context, order block tagging,
adjusted session windows, and a two-target (TP1/TP2) partial-close exit
to the liquidity sweep + break-of-structure + retest entry model.

This is a NEW, separate model from the v2 short/long models in
liquidity_sweep.py - it exists for comparison, not as a replacement.

Rules, as specified:
  - London session: 08:00-12:30 UK time.
  - US session (the active trading/entry window after London): 12:30-21:00
    UK time. Sweeps/entries are only considered within this window.
  - HTF trend: computed from 4H swing structure over the last 14 days.
    Downtrend = lower highs + lower lows. Uptrend = higher highs + higher
    lows. Otherwise "Range".
  - Order block: the most recent 4H candle opposite in colour to the HTF
    trend, immediately followed by an "impulsive" candle (range >= 1.5x
    the average range of the prior 10 bars) that closes in the trend
    direction. Tagged per trade for context, not used as a hard filter,
    since the backtest is partly about testing whether this context
    actually matters.
  - Entry mechanics (sweep, structure break, retest, 3-candle trigger)
    reuse the same logic as the v2 models - the same "CHoCH" concept,
    just now happening within the adjusted session window and reported
    alongside HTF context.
  - Exit: stop at the liquidity extreme (as before). TP1 = 2R - close 75%
    of the position and move stop to break-even. TP2 = the opposite
    session extreme (London low for shorts, London high for longs) for
    the remaining 25%.

KNOWN SIMPLIFICATION: the structure-break ("CHoCH") detection reuses the
existing 3-consecutive-candle heuristic from the v2 models rather than a
full fractal/pivot-based swing detector matching the LuxAlgo indicator's
exact internals. The HTF swing detection (for trend and order blocks)
DOES use real fractal pivots. This is disclosed so it's not mistaken for
a 1:1 replica of the indicator.
"""

import pandas as pd

from liquidity_sweep import (
    is_bearish_candle,
    is_bullish_candle,
    detect_bearish_bos_v2,
    detect_bullish_bos_v2,
    find_short_retest_entry_v2,
    find_long_retest_entry_v2,
)

LONDON_START = "08:00"
LONDON_END = "12:30"
US_SESSION_END = "21:00"

HTF_LOOKBACK_DAYS = 14
FRACTAL_WING = 2          # bars each side for a swing pivot
IMPULSE_MULTIPLIER = 1.5  # candle range vs recent average to count as "impulsive"
IMPULSE_LOOKBACK = 10


# ============================================================
# HIGHER-TIMEFRAME TREND (4H, fractal swings, 14-day lookback)
# ============================================================

def resample_to_4h(data_1h):
    if data_1h.empty:
        return pd.DataFrame()

    df = data_1h.set_index("Timestamp").sort_index()

    ohlc = df.resample("4h", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    ohlc = ohlc.reset_index()
    ohlc["Date"] = ohlc["Timestamp"].dt.date
    ohlc["Time"] = ohlc["Timestamp"].dt.time
    return ohlc


def find_fractal_swings(data_4h, wing=FRACTAL_WING):
    """
    Returns two lists of (index, price, timestamp): swing highs and swing
    lows, using a simple N-bar-each-side fractal (a bar is a swing high if
    its High is the max of the bars wing before and wing after it).
    """
    highs = []
    lows = []

    n = len(data_4h)
    for i in range(wing, n - wing):
        window = data_4h.iloc[i - wing: i + wing + 1]
        row = data_4h.iloc[i]

        if row["High"] == window["High"].max() and (window["High"] == row["High"]).sum() == 1:
            highs.append((i, row["High"], row["Timestamp"]))

        if row["Low"] == window["Low"].min() and (window["Low"] == row["Low"]).sum() == 1:
            lows.append((i, row["Low"], row["Timestamp"]))

    return highs, lows


def compute_htf_trend(data_4h_window):
    """
    Uptrend = last two swing highs rising AND last two swing lows rising.
    Downtrend = last two swing highs falling AND last two swing lows falling.
    Otherwise "Range".
    """
    if data_4h_window.empty or len(data_4h_window) < (FRACTAL_WING * 2 + 3):
        return "Range", None, None

    highs, lows = find_fractal_swings(data_4h_window)

    if len(highs) < 2 or len(lows) < 2:
        return "Range", None, None

    last_high, prev_high = highs[-1][1], highs[-2][1]
    last_low, prev_low = lows[-1][1], lows[-2][1]

    if last_high > prev_high and last_low > prev_low:
        return "Uptrend", highs, lows
    if last_high < prev_high and last_low < prev_low:
        return "Downtrend", highs, lows

    return "Range", highs, lows


# ============================================================
# ORDER BLOCK (simplified): last opposite-colour candle before
# an impulsive move in the trend direction
# ============================================================

def find_order_block(data_4h_window, trend):
    if trend not in ("Uptrend", "Downtrend") or len(data_4h_window) < IMPULSE_LOOKBACK + 2:
        return None

    ranges = (data_4h_window["High"] - data_4h_window["Low"]).reset_index(drop=True)
    rows = data_4h_window.reset_index(drop=True)

    for i in range(len(rows) - 2, IMPULSE_LOOKBACK, -1):
        impulse_row = rows.iloc[i]
        avg_range = ranges.iloc[i - IMPULSE_LOOKBACK: i].mean()
        impulse_range = ranges.iloc[i]

        is_impulsive = impulse_range >= IMPULSE_MULTIPLIER * avg_range
        closes_with_trend = (
            impulse_row["Close"] > impulse_row["Open"] if trend == "Uptrend"
            else impulse_row["Close"] < impulse_row["Open"]
        )

        if not (is_impulsive and closes_with_trend):
            continue

        ob_row = rows.iloc[i - 1]
        ob_is_opposite_colour = (
            is_bearish_candle(ob_row) if trend == "Uptrend"
            else is_bullish_candle(ob_row)
        )

        if ob_is_opposite_colour:
            return {
                "Timestamp": ob_row["Timestamp"],
                "Low": ob_row["Low"],
                "High": ob_row["High"],
            }

    return None


def order_block_retested(data_4h_window, order_block, as_of_timestamp):
    if order_block is None:
        return False

    later = data_4h_window[
        (data_4h_window["Timestamp"] > order_block["Timestamp"]) &
        (data_4h_window["Timestamp"] <= as_of_timestamp)
    ]

    if later.empty:
        return False

    touched = (later["Low"] <= order_block["High"]) & (later["High"] >= order_block["Low"])
    return bool(touched.any())


def htf_context_for_day(data_4h, as_of_timestamp):
    """
    Point-in-time HTF trend + order block context, using only 4H bars up
    to (not including) the day being evaluated - avoids lookahead bias.
    """
    cutoff = as_of_timestamp
    lookback_start = cutoff - pd.Timedelta(days=HTF_LOOKBACK_DAYS)

    window = data_4h[
        (data_4h["Timestamp"] >= lookback_start) & (data_4h["Timestamp"] < cutoff)
    ].copy()

    trend, highs, lows = compute_htf_trend(window)
    order_block = find_order_block(window, trend)
    retested = order_block_retested(window, order_block, cutoff)

    return {
        "trend": trend,
        "orderBlockFound": order_block is not None,
        "orderBlockRetested": retested,
    }


# ============================================================
# V3 TRADE SIMULATION: TP1 (2R, 75% close, move to BE) + TP2
# (opposite session extreme, remaining 25%)
# ============================================================

def simulate_short_trade_v3(trade_data, entry_level, stop_level, target_level):
    if trade_data.empty:
        return {"Outcome": "Unresolved", "Outcome Time": "", "R Multiple": None,
                "Plain English Result": "Entry triggered, but there was no data after entry."}

    risk_points = stop_level - entry_level
    reward_points = entry_level - target_level

    if risk_points <= 0:
        return {"Outcome": "Invalid Risk", "Outcome Time": "", "R Multiple": None,
                "Plain English Result": "Stop level is not above entry level."}
    if reward_points <= 0:
        return {"Outcome": "Invalid Reward", "Outcome Time": "", "R Multiple": None,
                "Plain English Result": "Target is not below entry level."}

    tp1_level = entry_level - (2.0 * risk_points)
    tp1_hit = False

    for _, row in trade_data.iterrows():
        stop_hit = row["High"] >= stop_level
        tp1_hit_now = row["Low"] <= tp1_level

        if not tp1_hit:
            if stop_hit and tp1_hit_now:
                return {"Outcome": "Ambiguous", "Outcome Time": row["Timestamp"], "R Multiple": None,
                        "Plain English Result": "Stop and TP1 touched in the same candle - can't confirm from OHLC."}
            if stop_hit:
                return {"Outcome": "Loss", "Outcome Time": row["Timestamp"], "R Multiple": -1.0,
                        "Plain English Result": "Stop-loss hit before TP1 (2R)."}
            if tp1_hit_now:
                tp1_hit = True
                tp1_time = row["Timestamp"]
                continue

        else:
            be_hit = row["High"] >= entry_level
            tp2_hit_now = row["Low"] <= target_level

            if be_hit and tp2_hit_now:
                return {"Outcome": "Ambiguous", "Outcome Time": row["Timestamp"], "R Multiple": None,
                        "Plain English Result": "Break-even stop and TP2 touched in the same candle."}
            if tp2_hit_now:
                r = (0.75 * 2.0) + (0.25 * ((entry_level - target_level) / risk_points))
                return {"Outcome": "Win (TP1+TP2)", "Outcome Time": row["Timestamp"], "R Multiple": round(r, 4),
                        "Plain English Result": "TP1 hit (75% closed, stop to BE), then TP2 hit on the runner."}
            if be_hit:
                r = 0.75 * 2.0
                return {"Outcome": "Win (TP1 only)", "Outcome Time": row["Timestamp"], "R Multiple": round(r, 4),
                        "Plain English Result": "TP1 hit (75% closed), runner stopped out at break-even."}

    if not tp1_hit:
        final_row = trade_data.iloc[-1]
        final_r = (entry_level - final_row["Close"]) / risk_points
        outcome = "Session Close Profit" if final_r > 0 else ("Session Close Flat" if abs(final_r) < 1e-9 else "Session Close Loss")
        return {"Outcome": outcome, "Outcome Time": final_row["Timestamp"], "R Multiple": round(final_r, 4),
                "Plain English Result": "Neither stop nor TP1 hit; closed at final available price."}

    final_row = trade_data.iloc[-1]
    runner_r = (entry_level - final_row["Close"]) / risk_points
    total_r = (0.75 * 2.0) + (0.25 * runner_r)
    return {"Outcome": "Win (TP1, runner open)", "Outcome Time": final_row["Timestamp"], "R Multiple": round(total_r, 4),
            "Plain English Result": "TP1 hit; runner closed at final available price (neither BE nor TP2 reached)."}


def simulate_long_trade_v3(trade_data, entry_level, stop_level, target_level):
    if trade_data.empty:
        return {"Outcome": "Unresolved", "Outcome Time": "", "R Multiple": None,
                "Plain English Result": "Entry triggered, but there was no data after entry."}

    risk_points = entry_level - stop_level
    reward_points = target_level - entry_level

    if risk_points <= 0:
        return {"Outcome": "Invalid Risk", "Outcome Time": "", "R Multiple": None,
                "Plain English Result": "Stop level is not below entry level."}
    if reward_points <= 0:
        return {"Outcome": "Invalid Reward", "Outcome Time": "", "R Multiple": None,
                "Plain English Result": "Target is not above entry level."}

    tp1_level = entry_level + (2.0 * risk_points)
    tp1_hit = False

    for _, row in trade_data.iterrows():
        stop_hit = row["Low"] <= stop_level
        tp1_hit_now = row["High"] >= tp1_level

        if not tp1_hit:
            if stop_hit and tp1_hit_now:
                return {"Outcome": "Ambiguous", "Outcome Time": row["Timestamp"], "R Multiple": None,
                        "Plain English Result": "Stop and TP1 touched in the same candle - can't confirm from OHLC."}
            if stop_hit:
                return {"Outcome": "Loss", "Outcome Time": row["Timestamp"], "R Multiple": -1.0,
                        "Plain English Result": "Stop-loss hit before TP1 (2R)."}
            if tp1_hit_now:
                tp1_hit = True
                continue

        else:
            be_hit = row["Low"] <= entry_level
            tp2_hit_now = row["High"] >= target_level

            if be_hit and tp2_hit_now:
                return {"Outcome": "Ambiguous", "Outcome Time": row["Timestamp"], "R Multiple": None,
                        "Plain English Result": "Break-even stop and TP2 touched in the same candle."}
            if tp2_hit_now:
                r = (0.75 * 2.0) + (0.25 * ((target_level - entry_level) / risk_points))
                return {"Outcome": "Win (TP1+TP2)", "Outcome Time": row["Timestamp"], "R Multiple": round(r, 4),
                        "Plain English Result": "TP1 hit (75% closed, stop to BE), then TP2 hit on the runner."}
            if be_hit:
                r = 0.75 * 2.0
                return {"Outcome": "Win (TP1 only)", "Outcome Time": row["Timestamp"], "R Multiple": round(r, 4),
                        "Plain English Result": "TP1 hit (75% closed), runner stopped out at break-even."}

    if not tp1_hit:
        final_row = trade_data.iloc[-1]
        final_r = (final_row["Close"] - entry_level) / risk_points
        outcome = "Session Close Profit" if final_r > 0 else ("Session Close Flat" if abs(final_r) < 1e-9 else "Session Close Loss")
        return {"Outcome": outcome, "Outcome Time": final_row["Timestamp"], "R Multiple": round(final_r, 4),
                "Plain English Result": "Neither stop nor TP1 hit; closed at final available price."}

    final_row = trade_data.iloc[-1]
    runner_r = (final_row["Close"] - entry_level) / risk_points
    total_r = (0.75 * 2.0) + (0.25 * runner_r)
    return {"Outcome": "Win (TP1, runner open)", "Outcome Time": final_row["Timestamp"], "R Multiple": round(total_r, 4),
            "Plain English Result": "TP1 hit; runner closed at final available price (neither BE nor TP2 reached)."}


# ============================================================
# FULL V3 DETECTION: session window + HTF context + entry + v3 exit
# ============================================================

def detect_v3_trade_entries(data_15m, data_5m, data_4h, direction, instrument="US30"):
    """
    direction: "Short" or "Long"
    instrument: label only (e.g. "US30", "US500") - tagged onto each row so
    results from multiple instruments can be pooled and still tell apart.
    """
    if data_15m.empty or data_5m.empty:
        return pd.DataFrame()

    results = []

    london_start_time = pd.to_datetime(LONDON_START).time()
    london_end_time = pd.to_datetime(LONDON_END).time()
    window_end_time = pd.to_datetime(US_SESSION_END).time()

    for trading_date, day_data_15m in data_15m.groupby("Date"):
        day_data_15m = day_data_15m.sort_values("Timestamp").copy()
        day_data_5m = data_5m[data_5m["Date"] == trading_date].sort_values("Timestamp").copy()

        if day_data_5m.empty:
            continue

        london_session = day_data_15m[
            (day_data_15m["Time"] >= london_start_time) &
            (day_data_15m["Time"] <= london_end_time)
        ]
        post_london_15m = day_data_15m[
            (day_data_15m["Time"] > london_end_time) &
            (day_data_15m["Time"] <= window_end_time)
        ]

        if london_session.empty or post_london_15m.empty:
            continue

        london_high = london_session["High"].max()
        london_low = london_session["Low"].min()

        day_start_ts = pd.Timestamp(london_session["Timestamp"].iloc[0])
        context = htf_context_for_day(data_4h, day_start_ts) if not data_4h.empty else \
            {"trend": "Unknown", "orderBlockFound": False, "orderBlockRetested": False}

        base_row = {
            "Date": trading_date,
            "Instrument": instrument,
            "Direction": direction,
            "HTF Trend": context["trend"],
            "Aligned With HTF": (
                (context["trend"] == "Downtrend" and direction == "Short") or
                (context["trend"] == "Uptrend" and direction == "Long")
            ),
            "Order Block Found": context["orderBlockFound"],
            "Order Block Retested": context["orderBlockRetested"],
            "London High": london_high,
            "London Low": london_low,
        }

        if direction == "Short":
            sweep_rows = post_london_15m[post_london_15m["High"] > london_high]
        else:
            sweep_rows = post_london_15m[post_london_15m["Low"] < london_low]

        if sweep_rows.empty:
            results.append({**base_row, "Sweep Time": "", "BOS Time": "", "Entry Time": "",
                             "Entry Level": None, "Stop Level": None, "Target Level": None,
                             "Entry Triggered": False, "Outcome": "No Sweep", "Outcome Time": "",
                             "R Multiple": None,
                             "Plain English Result": "No liquidity sweep within the session window."})
            continue

        sweep_row = sweep_rows.iloc[0]
        sweep_time = sweep_row["Timestamp"]

        data_after_sweep_5m = day_data_5m[
            (day_data_5m["Timestamp"] > sweep_time) &
            (day_data_5m["Time"] <= window_end_time)
        ].copy()

        bos = detect_bearish_bos_v2(data_after_sweep_5m) if direction == "Short" \
            else detect_bullish_bos_v2(data_after_sweep_5m)

        if bos is None:
            results.append({**base_row, "Sweep Time": sweep_time, "BOS Time": "", "Entry Time": "",
                             "Entry Level": None, "Stop Level": None, "Target Level": None,
                             "Entry Triggered": False, "Outcome": "No CHoCH", "Outcome Time": "",
                             "R Multiple": None,
                             "Plain English Result": "Swept, but no confirmed CHoCH within the window."})
            continue

        bos_time = bos["BOS Time"]
        pre_bos_5m = day_data_5m[
            (day_data_5m["Timestamp"] >= sweep_time) & (day_data_5m["Timestamp"] <= bos_time)
        ]

        if direction == "Short":
            liquidity_level = max(sweep_row["High"], pre_bos_5m["High"].max()) if not pre_bos_5m.empty else sweep_row["High"]
        else:
            liquidity_level = min(sweep_row["Low"], pre_bos_5m["Low"].min()) if not pre_bos_5m.empty else sweep_row["Low"]

        data_after_bos_5m = day_data_5m[
            (day_data_5m["Timestamp"] > bos_time) & (day_data_5m["Time"] <= window_end_time)
        ].copy()

        entry = find_short_retest_entry_v2(data_after_bos_5m, london_high, liquidity_level) if direction == "Short" \
            else find_long_retest_entry_v2(data_after_bos_5m, london_low, liquidity_level)

        if not entry["Entry Triggered"]:
            results.append({**base_row, "Sweep Time": sweep_time, "BOS Time": bos_time, "Entry Time": "",
                             "Entry Level": None, "Stop Level": liquidity_level, "Target Level": None,
                             "Entry Triggered": False, "Outcome": "No Entry", "Outcome Time": "",
                             "R Multiple": None, "Plain English Result": entry["Entry Failure Reason"]})
            continue

        entry_level = entry["Entry Level"]
        entry_time = entry["Entry Time"]
        stop_level = liquidity_level
        target_level = london_low if direction == "Short" else london_high

        trade_data = day_data_5m[
            (day_data_5m["Timestamp"] > entry_time) & (day_data_5m["Time"] <= window_end_time)
        ].copy()

        sim = simulate_short_trade_v3(trade_data, entry_level, stop_level, target_level) if direction == "Short" \
            else simulate_long_trade_v3(trade_data, entry_level, stop_level, target_level)

        results.append({
            **base_row,
            "Sweep Time": sweep_time, "BOS Time": bos_time, "Entry Time": entry_time,
            "Entry Level": entry_level, "Stop Level": stop_level, "Target Level": target_level,
            "Entry Triggered": True, "Outcome": sim["Outcome"], "Outcome Time": sim["Outcome Time"],
            "R Multiple": sim["R Multiple"], "Plain English Result": sim["Plain English Result"],
        })

    table = pd.DataFrame(results)

    if not table.empty:
        for col in ["Sweep Time", "BOS Time", "Entry Time", "Outcome Time"]:
            table[col] = pd.to_datetime(table[col], errors="coerce").dt.strftime("%H:%M").fillna("")

    return table
