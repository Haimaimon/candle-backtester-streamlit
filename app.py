#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from typing import List

import streamlit as st
import pandas as pd

from candle_backtester import (
    PATTERN_FUNCS,
    summarize_stats,
    BacktestStats,
    PatternBacktester,
    load_ohlc,
)

st.set_page_config(
    page_title="Candle Pattern Backtester",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Candle Pattern Backtester – LONG only")

# --- Session State init ---
if "trades_df" not in st.session_state:
    st.session_state["trades_df"] = None
if "summary_df" not in st.session_state:
    st.session_state["summary_df"] = None

st.sidebar.header("הגדרות בק-טסט")

# --- Sidebar controls ---

default_tickers = ["NU"]
tickers_input = st.sidebar.text_input(
    "Tickers (מופרדים בפסיק)", value=",".join(default_tickers)
)
tickers: List[str] = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

interval = st.sidebar.selectbox(
    "Interval (Timeframe)",
    options=["5m", "15m", "30m", "1h", "4h", "1d"],
    index=3,  # 1h
)

period = st.sidebar.selectbox(
    "Period (History)",
    options=["1mo", "3mo", "6mo", "1y"],
    index=2,  # 6mo
)

all_patterns = list(PATTERN_FUNCS.keys())
patterns_selected = st.sidebar.multiselect(
    "Patterns (אם ריק – יריץ על כולם)",
    options=all_patterns,
    default=all_patterns,
)

rr = st.sidebar.slider(
    "Risk/Reward (TP = R × SL)", min_value=1.0, max_value=4.0, step=0.5, value=2.0
)
max_bars = st.sidebar.slider(
    "Max bars in trade", min_value=5, max_value=50, step=1, value=10
)

st.sidebar.write("---")
run_button = st.sidebar.button("🚀 Run Backtest")


# --- Main area ---

# אם לוחצים על Run Backtest – מריצים בק-טסט ומעדכנים את ה-session_state
if run_button:
    if not tickers:
        st.error("לא הוזנו טיקרים.")
    else:
        backtester = PatternBacktester(
            rr=rr,
            max_bars_in_trade=max_bars,
            use_trend_filter=True,
            ma_window=50,
        )

        all_stats: List[BacktestStats] = []
        trades_rows = []

        progress = st.progress(0, text="מריץ בק-טסט...")

        total_steps = len(tickers) * max(
            1, len(patterns_selected) if patterns_selected else len(all_patterns)
        )
        curr_step = 0

        for symbol in tickers:
            st.write(f"### {symbol} – interval={interval}, period={period}")
            try:
                df = load_ohlc(symbol, interval=interval, period=period)
            except Exception as e:
                st.error(f"לא ניתן לטעון נתונים עבור {symbol}: {e}")
                continue

            this_patterns = patterns_selected if patterns_selected else all_patterns

            for pattern in this_patterns:
                stats = backtester.backtest_symbol_pattern(symbol, df, pattern)
                all_stats.append(stats)

                st.write(
                    f"- **{pattern}**: trades={stats.num_trades}, "
                    f"win_rate={stats.win_rate*100:.1f}%, "
                    f"avg_R={stats.avg_r:.2f}, total_R={stats.total_r:.2f}"
                )

                # נבנה גם את רשימת העסקאות הגולמית
                for t in stats.trades:
                    trades_rows.append(
                        {
                            "symbol": t.symbol,
                            "pattern": t.pattern,
                            "entry_time": t.entry_time,
                            "exit_time": t.exit_time,
                            "direction": t.direction,
                            "entry": t.entry,
                            "stop": t.stop,
                            "target": t.target,
                            "exit_price": t.exit_price,
                            "r_multiple": t.r_multiple,
                            "bars_in_trade": t.bars_in_trade,
                        }
                    )

                curr_step += 1
                progress.progress(
                    min(curr_step / total_steps, 1.0),
                    text=f"מריץ בק-טסט... ({curr_step}/{total_steps})",
                )

        if not all_stats:
            st.warning("לא נמצאו עסקאות.")
            # ננקה את ה-session_state כדי לא להציג נתונים ישנים
            st.session_state["trades_df"] = None
            st.session_state["summary_df"] = None
        else:
            summary_df = summarize_stats(all_stats)
            trades_df = pd.DataFrame(trades_rows) if trades_rows else pd.DataFrame()

            # ננקה timezone כדי להימנע מבעיות ב-Excel ובווידג'טים
            if not trades_df.empty:
                for col in ["entry_time", "exit_time"]:
                    if col in trades_df.columns and pd.api.types.is_datetime64_any_dtype(
                        trades_df[col]
                    ):
                        trades_df[col] = trades_df[col].dt.tz_localize(None)

            # נשמור ב-session_state כדי להשתמש בטאבים גם בריצות הבאות
            st.session_state["trades_df"] = trades_df
            st.session_state["summary_df"] = summary_df


# בשלב הזה – תמיד ניקח את הנתונים מה-session_state
trades_df = st.session_state.get("trades_df")
summary_df = st.session_state.get("summary_df")

# אם אין עדיין נתונים – נציג הודעה בלבד
if trades_df is None or summary_df is None or trades_df.empty:
    st.info("בחר מניות וטיימפריים בצד שמאל ולחץ על **Run Backtest**.")
else:
    st.success("הבק-טסט הושלם ✅")

    # ========= Tabs: Page 1 & Page 2 =========
    tab1, tab2 = st.tabs(
        ["📊 תוצאות בסיסיות", "🧠 ניתוח מתקדם / אסטרטגיה משופרת"]
    )

    # -------- TAB 1: תוצאות בסיסיות --------
    with tab1:
        st.subheader("סיכום לפי מניה + דפוס")
        st.dataframe(summary_df, use_container_width=True)

        if trades_df.empty:
            st.info("אין עסקאות להצגה.")
        else:
            st.subheader("כל העסקאות")
            st.dataframe(trades_df, use_container_width=True, height=400)

            # Download CSV
            csv_buffer = io.StringIO()
            trades_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ הורדת עסקאות (CSV)",
                data=csv_buffer.getvalue(),
                file_name="trades_streamlit.csv",
                mime="text/csv",
            )

            # Download Excel (summary + trades)
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                trades_df.to_excel(writer, sheet_name="trades", index=False)
                summary_df.to_excel(writer, sheet_name="summary", index=False)
            st.download_button(
                label="⬇️ הורדת תוצאות (Excel)",
                data=excel_buffer.getvalue(),
                file_name="backtest_results_streamlit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # -------- TAB 2: ניתוח מתקדם / אסטרטגיה משופרת --------
    with tab2:
        st.subheader("פילטרים לאסטרטגיה משופרת")

        if trades_df.empty:
            st.info("אין עסקאות לניתוח.")
        else:
            # פילטרים: תאריך, דפוס, bars_in_trade
            min_date = trades_df["entry_time"].min().date()
            max_date = trades_df["entry_time"].max().date()

            date_range = st.date_input(
                "טווח תאריכים (לפי entry_time)",
                value=[min_date, max_date],
            )

            patterns_options = sorted(trades_df["pattern"].unique())
            patterns_filter = st.multiselect(
                "בחר דפוסים",
                options=patterns_options,
                default=patterns_options,
            )

            min_bars = int(trades_df["bars_in_trade"].min())
            max_bars_total = int(trades_df["bars_in_trade"].max())
            max_bars_filter = st.slider(
                "מספר מקסימלי של נרות בעסקה (bars_in_trade)",
                min_value=min_bars,
                max_value=max_bars_total,
                value=max_bars_total,
            )

            compute_btn = st.button("💡 חישוב אסטרטגיה משופרת")

            if compute_btn:
                df_filtered = trades_df.copy()

                # סינון לפי תאריכים
                if isinstance(date_range, list) and len(date_range) == 2:
                    start = pd.to_datetime(date_range[0])
                    end = pd.to_datetime(date_range[1])
                    df_filtered = df_filtered[
                        (df_filtered["entry_time"] >= start)
                        & (df_filtered["entry_time"] <= end)
                    ]

                # סינון לפי דפוסים
                if patterns_filter:
                    df_filtered = df_filtered[
                        df_filtered["pattern"].isin(patterns_filter)
                    ]

                # סינון לפי bars_in_trade
                df_filtered = df_filtered[
                    df_filtered["bars_in_trade"] <= max_bars_filter
                ]

                if df_filtered.empty:
                    st.warning("לא נמצאו עסקאות לאחר הפילטרים.")
                else:
                    st.markdown("### תוצאות האסטרטגיה המשופרת")

                    num_trades = len(df_filtered)
                    wins = (df_filtered["r_multiple"] > 0).sum()
                    win_rate = wins / num_trades * 100.0
                    avg_R = df_filtered["r_multiple"].mean()
                    total_R = df_filtered["r_multiple"].sum()

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("מספר עסקאות", num_trades)
                    col2.metric("אחוז הצלחה", f"{win_rate:.1f}%")
                    col3.metric("R ממוצע לעסקה", f"{avg_R:.2f}")
                    col4.metric("R כולל", f"{total_R:.2f}")

                    # Equity Curve
                    st.subheader("Equity Curve – R מצטבר")
                    df_equity = df_filtered.sort_values("exit_time").copy()
                    df_equity["cum_R"] = df_equity["r_multiple"].cumsum()
                    st.line_chart(df_equity.set_index("exit_time")["cum_R"])

                    # גרף R לכל עסקה על ציר זמן
                    st.subheader("R לכל עסקה על ציר זמן")
                    st.bar_chart(df_equity.set_index("exit_time")["r_multiple"])

                    # הצגת העסקאות המסוננות
                    st.subheader("רשימת העסקאות לאחר הפילטרים")
                    st.dataframe(
                        df_filtered,
                        use_container_width=True,
                        height=350,
                    )
