#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import logging
import time
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

# הגדרת logging - מונע דפליקציה
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # אם יש הגדרות קודמות, נתקע אותן
)
logger = logging.getLogger(__name__)
logger.propagate = False  # מונע העברה ל-root logger (מונע דפליקציה)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
st.sidebar.subheader("🎯 Pattern Validator (12 Filters)")

use_validator = st.sidebar.checkbox(
    "השתמש ב-Pattern Validator (12 פילטרים)", value=True
)

min_pattern_score = 0.0
require_entry_trigger = True
require_volume_confirmation = True
max_bars_before_pattern = 20

if use_validator:
    min_pattern_score = st.sidebar.slider(
        "ציון מינימלי לדפוס (0-100)", min_value=0.0, max_value=100.0, step=5.0, value=40.0
    )
    require_entry_trigger = st.sidebar.checkbox(
        "דרוש Entry Trigger", value=True
    )
    require_volume_confirmation = st.sidebar.checkbox(
        "דרוש Volume Confirmation", value=True
    )
    max_bars_before_pattern = st.sidebar.slider(
        "Max bars before pattern", min_value=5, max_value=50, step=5, value=20
    )

st.sidebar.write("---")
run_button = st.sidebar.button("🚀 Run Backtest")


# --- Main area ---

# אם לוחצים על Run Backtest – מריצים בק-טסט ומעדכנים את ה-session_state
if run_button:
    if not tickers:
        st.error("לא הוזנו טיקרים.")
    else:
        logger.info("=" * 60)
        logger.info("Starting backtest session")
        logger.info(f"Tickers: {tickers}")
        logger.info(f"Interval: {interval}, Period: {period}")
        logger.info(f"Patterns: {patterns_selected if patterns_selected else 'All'}")
        logger.info(f"Validator: {use_validator}, Min Score: {min_pattern_score}")
        logger.info("=" * 60)
        
        overall_start_time = time.time()
        
        backtester = PatternBacktester(
            rr=rr,
            max_bars_in_trade=max_bars,
            use_trend_filter=True,
            ma_window=50,
            use_pattern_validator=use_validator,
            min_pattern_score=min_pattern_score,
            require_entry_trigger=require_entry_trigger,
            require_volume_confirmation=require_volume_confirmation,
            max_bars_before_pattern=max_bars_before_pattern,
        )

        all_stats: List[BacktestStats] = []
        trades_rows = []

        progress = st.progress(0, text="מריץ בק-טסט...")

        total_steps = len(tickers) * max(
            1, len(patterns_selected) if patterns_selected else len(all_patterns)
        )
        curr_step = 0

        for symbol in tickers:
            symbol_start_time = time.time()
            logger.info(f"\n--- Processing {symbol} ---")
            st.write(f"### {symbol} – interval={interval}, period={period}")
            try:
                logger.info(f"Loading data for {symbol}...")
                load_start = time.time()
                df = load_ohlc(symbol, interval=interval, period=period)
                load_time = time.time() - load_start
                logger.info(f"Data loaded: {len(df)} candles in {load_time:.2f} seconds")
            except Exception as e:
                error_msg = f"לא ניתן לטעון נתונים עבור {symbol}: {e}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                continue

            this_patterns = patterns_selected if patterns_selected else all_patterns
            logger.info(f"Testing {len(this_patterns)} patterns for {symbol}")

            for pattern in this_patterns:
                pattern_start = time.time()
                logger.info(f"Testing pattern: {pattern}")
                
                try:
                    stats = backtester.backtest_symbol_pattern(symbol, df, pattern)
                    pattern_time = time.time() - pattern_start
                    all_stats.append(stats)

                    logger.info(f"Pattern {pattern} completed in {pattern_time:.2f}s: "
                              f"trades={stats.num_trades}, win_rate={stats.win_rate*100:.1f}%, "
                              f"avg_R={stats.avg_r:.2f}, total_R={stats.total_r:.2f}")

                    st.write(
                        f"- **{pattern}**: trades={stats.num_trades}, "
                        f"win_rate={stats.win_rate*100:.1f}%, "
                        f"avg_R={stats.avg_r:.2f}, total_R={stats.total_r:.2f}"
                    )
                except Exception as e:
                    error_msg = f"Error testing pattern {pattern} for {symbol}: {e}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    continue

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
                            "volume_before": t.volume_before,
                            "volume_after": t.volume_after,
                            "is_trending": t.is_trending,
                            "is_support_zone": t.is_support_zone,
                            "rsi_divergence": t.rsi_divergence,
                            "candle_range_strength": t.candle_range_strength,
                            "pattern_strength_score": t.pattern_strength_score,
                        }
                    )

                curr_step += 1
                progress.progress(
                    min(curr_step / total_steps, 1.0),
                    text=f"מריץ בק-טסט... ({curr_step}/{total_steps})",
                )
            
            symbol_time = time.time() - symbol_start_time
            logger.info(f"{symbol} completed in {symbol_time:.2f} seconds")

        overall_time = time.time() - overall_start_time
        total_trades = sum(stats.num_trades for stats in all_stats)
        
        logger.info("=" * 60)
        logger.info("Backtest session complete")
        logger.info(f"Total symbols processed: {len(tickers)}")
        logger.info(f"Total patterns tested: {len(all_stats)}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total elapsed time: {overall_time:.2f} seconds")
        logger.info("=" * 60)
        
        if not all_stats:
            st.warning("לא נמצאו עסקאות.")
            logger.warning("No trades found in backtest")
            # ננקה את ה-session_state כדי לא להציג נתונים ישנים
            st.session_state["trades_df"] = None
            st.session_state["summary_df"] = None
        else:
            logger.info("Creating summary and trades dataframe...")
            summary_df = summarize_stats(all_stats)
            trades_df = pd.DataFrame(trades_rows) if trades_rows else pd.DataFrame()
            logger.info(f"Summary created: {len(summary_df)} rows, Trades: {len(trades_df)} rows")

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
            # בדיקה אם יש שדות חדשים
            has_pattern_score = "pattern_strength_score" in trades_df.columns
            
            if has_pattern_score:
                # ========= Heatmap לפי Pattern Strength Score =========
                st.subheader("🔥 Heatmap של הצלחה לפי Pattern Strength Score")
                
                # יצירת bins לציון
                score_bins = [0, 30, 50, 70, 85, 100]
                trades_df_temp = trades_df.copy()
                trades_df_temp["score_bin"] = pd.cut(
                    trades_df_temp["pattern_strength_score"],
                    bins=score_bins,
                    labels=["0-30", "30-50", "50-70", "70-85", "85-100"],
                    include_lowest=True
                )
                
                # חישוב win rate לפי דפוס וציון
                heatmap_data = []
                for pattern in trades_df_temp["pattern"].unique():
                    for score_bin in trades_df_temp["score_bin"].cat.categories:
                        subset = trades_df_temp[
                            (trades_df_temp["pattern"] == pattern) &
                            (trades_df_temp["score_bin"] == score_bin)
                        ]
                        if len(subset) > 0:
                            win_rate = (subset["r_multiple"] > 0).sum() / len(subset) * 100
                            avg_r = subset["r_multiple"].mean()
                            num_trades = len(subset)
                            heatmap_data.append({
                                "pattern": pattern,
                                "score_range": str(score_bin),
                                "win_rate": win_rate,
                                "avg_r": avg_r,
                                "num_trades": num_trades,
                            })
                
                if heatmap_data:
                    heatmap_df = pd.DataFrame(heatmap_data)
                    
                    # יצירת pivot table ל-heatmap
                    pivot_winrate = heatmap_df.pivot(
                        index="pattern",
                        columns="score_range",
                        values="win_rate"
                    ).fillna(0)
                    
                    pivot_avg_r = heatmap_df.pivot(
                        index="pattern",
                        columns="score_range",
                        values="avg_r"
                    ).fillna(0)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Win Rate (%) לפי Pattern ו-Score**")
                        try:
                            # ניסיון להציג עם background gradient (דורש matplotlib)
                            styled_df = pivot_winrate.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=100)
                            st.dataframe(styled_df, use_container_width=True)
                        except ImportError:
                            # אם matplotlib לא מותקן, הצג ללא styling
                            st.dataframe(pivot_winrate, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Average R לפי Pattern ו-Score**")
                        try:
                            # ניסיון להציג עם background gradient (דורש matplotlib)
                            styled_df = pivot_avg_r.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=2)
                            st.dataframe(styled_df, use_container_width=True)
                        except ImportError:
                            # אם matplotlib לא מותקן, הצג ללא styling
                            st.dataframe(pivot_avg_r, use_container_width=True)
                    
                    # גרף הצלחה לפי ציון
                    st.markdown("### Win Rate לפי Pattern Strength Score")
                    score_summary = trades_df_temp.groupby("score_bin").agg({
                        "r_multiple": lambda x: (x > 0).sum() / len(x) * 100,
                        "pattern_strength_score": "count"
                    }).rename(columns={"r_multiple": "win_rate", "pattern_strength_score": "num_trades"})
                    st.bar_chart(score_summary["win_rate"])
                
                # ========= Optimal Conditions Finder =========
                st.subheader("🎯 Optimal Conditions Finder")
                st.markdown("**מציאת התנאים האופטימליים שהביאו הכי הרבה רווח**")
                
                # ניתוח לפי תנאים שונים
                conditions_analysis = []
                
                # ניתוח לפי Support Zone
                if "is_support_zone" in trades_df.columns:
                    for support in [True, False]:
                        subset = trades_df[trades_df["is_support_zone"] == support]
                        if len(subset) > 0:
                            win_rate = (subset["r_multiple"] > 0).sum() / len(subset) * 100
                            avg_r = subset["r_multiple"].mean()
                            conditions_analysis.append({
                                "condition": f"Support Zone: {support}",
                                "num_trades": len(subset),
                                "win_rate": win_rate,
                                "avg_r": avg_r,
                                "total_r": subset["r_multiple"].sum(),
                            })
                
                # ניתוח לפי RSI Divergence
                if "rsi_divergence" in trades_df.columns:
                    for rsi_div in [True, False]:
                        subset = trades_df[trades_df["rsi_divergence"] == rsi_div]
                        if len(subset) > 0:
                            win_rate = (subset["r_multiple"] > 0).sum() / len(subset) * 100
                            avg_r = subset["r_multiple"].mean()
                            conditions_analysis.append({
                                "condition": f"RSI Divergence: {rsi_div}",
                                "num_trades": len(subset),
                                "win_rate": win_rate,
                                "avg_r": avg_r,
                                "total_r": subset["r_multiple"].sum(),
                            })
                
                # ניתוח לפי Trend
                if "is_trending" in trades_df.columns:
                    for trending in [True, False]:
                        subset = trades_df[trades_df["is_trending"] == trending]
                        if len(subset) > 0:
                            win_rate = (subset["r_multiple"] > 0).sum() / len(subset) * 100
                            avg_r = subset["r_multiple"].mean()
                            conditions_analysis.append({
                                "condition": f"Trending: {trending}",
                                "num_trades": len(subset),
                                "win_rate": win_rate,
                                "avg_r": avg_r,
                                "total_r": subset["r_multiple"].sum(),
                            })
                
                # ניתוח לפי ציון גבוה (>= 70)
                if has_pattern_score:
                    for min_score in [70, 85]:
                        subset = trades_df[trades_df["pattern_strength_score"] >= min_score]
                        if len(subset) > 0:
                            win_rate = (subset["r_multiple"] > 0).sum() / len(subset) * 100
                            avg_r = subset["r_multiple"].mean()
                            conditions_analysis.append({
                                "condition": f"Pattern Score >= {min_score}",
                                "num_trades": len(subset),
                                "win_rate": win_rate,
                                "avg_r": avg_r,
                                "total_r": subset["r_multiple"].sum(),
                            })
                
                if conditions_analysis:
                    conditions_df = pd.DataFrame(conditions_analysis)
                    conditions_df = conditions_df.sort_values("avg_r", ascending=False)
                    st.dataframe(conditions_df, use_container_width=True)
                    
                    # הצגת התנאים הטובים ביותר
                    st.markdown("### 🏆 התנאים הטובים ביותר")
                    top_conditions = conditions_df.head(5)
                    for idx, row in top_conditions.iterrows():
                        st.markdown(
                            f"- **{row['condition']}**: "
                            f"Win Rate={row['win_rate']:.1f}%, "
                            f"Avg R={row['avg_r']:.2f}, "
                            f"Total R={row['total_r']:.2f}, "
                            f"Trades={int(row['num_trades'])}"
                        )
                
                # ========= סינון לפי ציון מינימלי =========
                st.markdown("---")
                st.subheader("📊 סקירת טריידים רק בציון גבוה")
                
                min_score_filter = st.slider(
                    "ציון מינימלי להצגה",
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    value=70.0 if has_pattern_score else 0.0,
                )
                
                high_score_trades = trades_df[
                    trades_df["pattern_strength_score"] >= min_score_filter
                ] if has_pattern_score else trades_df
                
                if not high_score_trades.empty:
                    st.markdown(f"**נמצאו {len(high_score_trades)} עסקאות עם ציון >= {min_score_filter}**")
                    
                    num_trades_high = len(high_score_trades)
                    wins_high = (high_score_trades["r_multiple"] > 0).sum()
                    win_rate_high = wins_high / num_trades_high * 100.0
                    avg_R_high = high_score_trades["r_multiple"].mean()
                    total_R_high = high_score_trades["r_multiple"].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("מספר עסקאות", num_trades_high)
                    col2.metric("אחוז הצלחה", f"{win_rate_high:.1f}%")
                    col3.metric("R ממוצע", f"{avg_R_high:.2f}")
                    col4.metric("R כולל", f"{total_R_high:.2f}")
                    
                    st.dataframe(
                        high_score_trades.sort_values("pattern_strength_score", ascending=False),
                        use_container_width=True,
                        height=400,
                    )
                else:
                    st.warning(f"לא נמצאו עסקאות עם ציון >= {min_score_filter}")
            
            # ========= המשך עם הפילטרים הקיימים =========
            # פילטרים: תאריך, דפוס, bars_in_trade
            st.markdown("---")
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
