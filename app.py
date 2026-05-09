import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from liquidity_sweep import (
    load_intraday_data,
    detect_short_trade_entries,
    summarise_short_trade_entries
)


st.set_page_config(
    page_title="Rand & Risk Quant Lab",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    .brand-card {
        border: 1px solid #262626;
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1.2rem;
        background: rgba(255,255,255,0.02);
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .brand-logo {
        width: 50px;
        height: 50px;
        border-radius: 14px;
        background: linear-gradient(135deg, #111111 0%, #1f1f1f 100%);
        color: #f0c36d;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: -0.03em;
        border: 1px solid #333333;
    }

    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }

    .brand-subtitle {
        margin: 0.15rem 0 0 0;
        color: #a8a8a8;
        font-size: 0.95rem;
    }

    .small-text {
        color: #b0b0b0;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .info-card {
        border: 1px solid #262626;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.02);
    }

    .footer-note {
        color: #8f8f8f;
        font-size: 0.85rem;
        margin-top: 2rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 650;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #9e9e9e;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FIXED SETTINGS
# ============================================================

MARKET_LABEL = "US30 Proxy - Dow Futures"
TICKER = "YM=F"
LONDON_START = "08:00"
LONDON_END = "13:00"

# Free Yahoo/yfinance intraday data is usually limited to around 60 calendar days.
INTRADAY_PERIOD = "60d"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_all_data():
    data_15m = load_intraday_data(
        ticker=TICKER,
        period=INTRADAY_PERIOD,
        interval="15m"
    )

    data_5m = load_intraday_data(
        ticker=TICKER,
        period=INTRADAY_PERIOD,
        interval="5m"
    )

    return data_15m, data_5m


data_15m, data_5m = load_all_data()

if data_15m.empty or data_5m.empty:
    st.error("No intraday data was loaded. Please try again.")
    st.stop()


short_trade_table = detect_short_trade_entries(
    data_15m=data_15m,
    data_5m=data_5m,
    london_start=LONDON_START,
    london_end=LONDON_END
)

short_trade_summary = summarise_short_trade_entries(short_trade_table)

if short_trade_table.empty:
    st.warning("No short entry model data was available.")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-card">
        <div class="brand-row">
            <div class="brand-logo">R&R</div>
            <div>
                <p class="brand-title">Rand & Risk Quant Lab</p>
                <p class="brand-subtitle">London range behaviour research and systematic US30 strategy testing.</p>
            </div>
        </div>
        <p class="small-text">
        I am building this project to test whether one of my discretionary US30 trading ideas can be turned into a rules-based quant system.
        Instead of relying on chart feel alone, I am using intraday data to test whether a London high liquidity event, bearish break of structure,
        retest and pullback entry can produce measurable results. The aim is not to claim I have found a finished trading system, but to build
        a transparent research process that shows how the strategy behaves under fixed rules.
        </p>
        <p class="small-text">
        Current model:
        London high liquidity event → strict bearish 5-minute BOS → retest of liquidity zone →
        3 bullish candle pullback entry → stop at liquidity high → target at London low.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL SETTINGS
# ============================================================

st.subheader("Model Settings")

set1, set2, set3, set4 = st.columns(4)

with set1:
    st.metric("Market", MARKET_LABEL)

with set2:
    st.metric("London Range", f"{LONDON_START}–{LONDON_END}")

with set3:
    st.metric("Data Used", "15m + 5m")

with set4:
    st.metric("Data Window", INTRADAY_PERIOD)


st.markdown(
    """
    <div class="info-card">
        <p class="small-text">
        <b>Note on sample size:</b> free Yahoo Finance intraday data is capped at roughly 60 calendar days.
        That usually translates into around 45–50 usable trading days in this model. A longer sample would require a local data archive or an external historical intraday dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STRATEGY METRICS
# ============================================================

st.subheader("Strategy Snapshot")

total_model_days = len(short_trade_table)
entry_triggered_count = int(short_trade_table["Entry Triggered"].sum())

profitable_trades = short_trade_table["Outcome"].isin(
    ["Win", "Session Close Profit"]
).sum()

losing_trades = short_trade_table["Outcome"].isin(
    ["Loss", "Session Close Loss"]
).sum()

flat_trades = short_trade_table["Outcome"].isin(
    ["Break-even", "Session Close Flat"]
).sum()

session_close_profit = (short_trade_table["Outcome"] == "Session Close Profit").sum()
session_close_loss = (short_trade_table["Outcome"] == "Session Close Loss").sum()
session_close_flat = (short_trade_table["Outcome"] == "Session Close Flat").sum()

no_bos = (short_trade_table["Outcome"] == "No BOS").sum()
no_entry = (short_trade_table["Outcome"] == "No Entry").sum()
no_sweep = (short_trade_table["Outcome"] == "No Sweep").sum()
ambiguous = (short_trade_table["Outcome"] == "Ambiguous").sum()

resolved_trades = profitable_trades + losing_trades + flat_trades

if resolved_trades > 0:
    model_win_rate = profitable_trades / resolved_trades * 100
else:
    model_win_rate = 0

average_r = short_trade_table["R Multiple"].dropna().mean()
total_r = short_trade_table["R Multiple"].dropna().sum()

if total_model_days > 0:
    entry_rate = entry_triggered_count / total_model_days * 100
else:
    entry_rate = 0


# Build executed trade curve data
executed_trades = short_trade_table[
    (short_trade_table["Entry Triggered"] == True) &
    (short_trade_table["R Multiple"].notna())
].copy()

if not executed_trades.empty:
    executed_trades["Date"] = pd.to_datetime(executed_trades["Date"])
    executed_trades = executed_trades.sort_values(["Date", "Entry Time"]).reset_index(drop=True)
    executed_trades["Trade Number"] = executed_trades.index + 1
    executed_trades["Cumulative R"] = executed_trades["R Multiple"].cumsum()
    executed_trades["Running Peak R"] = executed_trades["Cumulative R"].cummax()
    executed_trades["Drawdown R"] = executed_trades["Cumulative R"] - executed_trades["Running Peak R"]
    max_r_drawdown = executed_trades["Drawdown R"].min()

    r_series = executed_trades["R Multiple"].dropna()

    if len(r_series) > 1 and r_series.std() != 0:
        trade_sharpe = (r_series.mean() / r_series.std()) * (len(r_series) ** 0.5)
    else:
        trade_sharpe = 0
else:
    max_r_drawdown = 0
    trade_sharpe = 0


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model Days", f"{total_model_days}")

with m2:
    st.metric("Entries Triggered", f"{entry_triggered_count}")

with m3:
    st.metric("Entry Rate", f"{entry_rate:.2f}%")

with m4:
    st.metric("Model Win Rate", f"{model_win_rate:.2f}%")


m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric("Profitable Trades", f"{profitable_trades}")

with m6:
    st.metric("Losing Trades", f"{losing_trades}")

with m7:
    st.metric("Flat / Break-even", f"{flat_trades}")

with m8:
    if pd.isna(average_r):
        st.metric("Average R", "N/A")
    else:
        st.metric("Average R", f"{average_r:.2f}")


m9, m10, m11, m12 = st.columns(4)

with m9:
    st.metric("Total R", f"{total_r:.2f}")

with m10:
    st.metric("Trade Sharpe", f"{trade_sharpe:.2f}")

with m11:
    st.metric("Max R Drawdown", f"{max_r_drawdown:.2f}")

with m12:
    st.metric("No Entry / No Sweep", f"{no_entry + no_sweep}")


st.markdown(
    """
    <div class="info-card">
        <p class="small-text">
        <b>Trade Sharpe note:</b> this is calculated from the model's trade-by-trade R multiples rather than daily portfolio returns.
        It is useful as a rough consistency measure, but the sample is still small and should not be treated like a fully institutional Sharpe ratio.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# R-MULTIPLE EQUITY CURVE
# ============================================================

st.subheader("R-Multiple Equity Curve")

if executed_trades.empty:
    st.info("No executed trades with valid R multiples were available for the equity curve.")
else:
    equity_fig = go.Figure()

    equity_fig.add_trace(
        go.Scatter(
            x=executed_trades["Trade Number"],
            y=executed_trades["Cumulative R"],
            mode="lines+markers",
            name="Cumulative R",
            line=dict(width=3)
        )
    )

    equity_fig.add_trace(
        go.Scatter(
            x=executed_trades["Trade Number"],
            y=executed_trades["Running Peak R"],
            mode="lines",
            name="Running Peak",
            line=dict(width=1.5, dash="dot")
        )
    )

    equity_fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Trade Number",
        yaxis_title="Cumulative R",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(equity_fig, use_container_width=True)

    dd_fig = go.Figure()

    dd_fig.add_trace(
        go.Bar(
            x=executed_trades["Trade Number"],
            y=executed_trades["Drawdown R"],
            name="Drawdown R"
        )
    )

    dd_fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Trade Number",
        yaxis_title="Drawdown in R",
        template="plotly_white",
        showlegend=False
    )

    st.plotly_chart(dd_fig, use_container_width=True)


# ============================================================
# SESSION-CLOSE BREAKDOWN
# ============================================================

st.subheader("Session-Close Breakdown")

sc1, sc2, sc3, sc4 = st.columns(4)

with sc1:
    st.metric("Session Close Profit", f"{session_close_profit}")

with sc2:
    st.metric("Session Close Loss", f"{session_close_loss}")

with sc3:
    st.metric("Session Close Flat", f"{session_close_flat}")

with sc4:
    st.metric("Ambiguous", f"{ambiguous}")


# ============================================================
# OUTCOME SUMMARY
# ============================================================

st.subheader("Outcome Summary")

chart_col, table_col = st.columns([1.1, 0.9])

with chart_col:
    outcome_chart = go.Figure()

    outcome_chart.add_trace(
        go.Bar(
            x=short_trade_summary["Outcome"],
            y=short_trade_summary["Count"],
            text=short_trade_summary["Count"],
            textposition="outside"
        )
    )

    outcome_chart.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="",
        yaxis_title="Count",
        template="plotly_white",
        showlegend=False
    )

    st.plotly_chart(outcome_chart, use_container_width=True)

with table_col:
    st.dataframe(
        short_trade_summary,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# RECENT TRADES
# ============================================================

st.subheader("Recent Trade Log")

display_short_trades = short_trade_table[
    [
        "Date",
        "Sweep Time",
        "BOS Time",
        "Entry Time",
        "London High",
        "London Low",
        "Sweep High",
        "Entry Level",
        "Stop Level",
        "Target Level",
        "Outcome",
        "R Multiple",
        "Plain English Result"
    ]
].copy()

display_short_trades = display_short_trades.sort_values(
    by=["Date", "Entry Time"],
    ascending=[False, False]
).head(20)

st.dataframe(
    display_short_trades,
    hide_index=True,
    use_container_width=True
)


short_trade_csv = short_trade_table.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download full trade log as CSV",
    data=short_trade_csv,
    file_name="rand_and_risk_quant_lab_trade_log.csv",
    mime="text/csv"
)


# ============================================================
# NEXT STEP
# ============================================================

st.subheader("Next Best Upgrade")

st.markdown(
    """
    <div class="info-card">
        <p class="small-text">
        The next best upgrade is to add a <b>major U.S. news-day filter</b>, because my trading rule is to avoid CPI, FOMC,
        NFP and other high-impact U.S. macro days. After that, I should manually check the triggered trades against TradingView
        to make sure the model's BOS and retest logic matches how I would read the setup in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <p class="footer-note">
    Rand & Risk Quant Lab. Research project only. Not financial advice or a live trading signal service.
    </p>
    """,
    unsafe_allow_html=True
)