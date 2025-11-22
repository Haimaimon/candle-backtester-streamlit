#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Candle Pattern Backtester v2
- LONG only
- Supports multiple tickers, timeframe & period
- Detects bullish candlestick patterns and backtests them
- Exports trades to CSV/Excel
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# ========= Domain Models =========

@dataclass
class TradeResult:
    symbol: str
    pattern: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "LONG"
    entry: float
    stop: float
    target: float
    exit_price: float
    r_multiple: float
    bars_in_trade: int


@dataclass
class BacktestStats:
    symbol: str
    pattern: str
    trades: List[TradeResult]

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.r_multiple > 0)
        return wins / len(self.trades)

    @property
    def avg_r(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean([t.r_multiple for t in self.trades]))

    @property
    def total_r(self) -> float:
        return float(np.sum([t.r_multiple for t in self.trades]))


# ========= Data Loader =========

def load_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Load OHLCV data using yfinance.
    interval: "1h", "5m" וכו'
    period: "6mo", "1mo", "1y" ...
    דואג לטפל גם במקרה של MultiIndex בעמודות (כש-yfinance מחזיר ('Open','NU') וכו')
    """
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data for {symbol} ({interval}, {period})")

    # --- טיפול במקרה של MultiIndex בעמודות ---
    if isinstance(df.columns, pd.MultiIndex):
        # בדרך כלל ב-yfinance: רמת 0 = "Open/High/Low/Close/Volume",
        # רמת 1 = ticker (כמו 'NU')
        try:
            if symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
            else:
                df = df.xs(symbol, axis=1, level=0)
        except Exception as e:
            print("Columns from yfinance:", df.columns)
            raise RuntimeError(f"Failed to flatten MultiIndex columns for {symbol}") from e

    df = df.rename(columns=str.title)  # open -> Open וכו'
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    return df


# ========= Pattern Detection =========

def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    prev_bearish = c.shift(1) < o.shift(1)
    curr_bullish = c > o
    body_prev = (c.shift(1) - o.shift(1)).abs()
    body_curr = (c - o).abs()

    engulf = (o <= c.shift(1)) & (c >= o.shift(1))
    return prev_bearish & curr_bullish & engulf & (body_curr > body_prev * 0.8)


def is_hammer(df: pd.DataFrame, max_body_ratio: float = 0.3,
              min_lower_shadow_ratio: float = 2.0) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    range_ = h - l
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l

    small_body = body <= range_ * max_body_ratio
    long_lower = lower_shadow >= range_ * (min_lower_shadow_ratio / (1 + min_lower_shadow_ratio))
    short_upper = upper_shadow <= body

    prior_down = c.shift(1).rolling(3).mean() < c.shift(4).rolling(3).mean()
    return small_body & long_lower & short_upper & prior_down


def is_inverted_hammer(df: pd.DataFrame, max_body_ratio: float = 0.3,
                       min_upper_shadow_ratio: float = 2.0) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    range_ = h - l
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l

    small_body = body <= range_ * max_body_ratio
    long_upper = upper_shadow >= range_ * (min_upper_shadow_ratio / (1 + min_upper_shadow_ratio))
    short_lower = lower_shadow <= body
    prior_down = c.shift(1).rolling(3).mean() < c.shift(4).rolling(3).mean()
    return small_body & long_upper & short_lower & prior_down


def is_bullish_harami(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    prev_bearish = c.shift(1) < o.shift(1)
    curr_bullish = c > o

    body_prev_low = np.minimum(c.shift(1), o.shift(1))
    body_prev_high = np.maximum(c.shift(1), o.shift(1))
    body_curr_low = np.minimum(c, o)
    body_curr_high = np.maximum(c, o)

    inside = (body_curr_low > body_prev_low) & (body_curr_high < body_prev_high)

    return prev_bearish & curr_bullish & inside


def is_tweezer_bottom(df: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    o, c, l = df["Open"], df["Close"], df["Low"]
    low_eq = (l - l.shift(1)).abs() <= (l * tolerance)
    first_bearish = c.shift(1) < o.shift(1)
    second_bullish = c > o
    prior_down = c.shift(1) < c.shift(2)
    return low_eq & first_bearish & second_bullish & prior_down


def is_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]

    bull1 = c.shift(2) > o.shift(2)
    bull2 = c.shift(1) > o.shift(1)
    bull3 = c > o

    higher_close = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    small_wicks = ((h - c) < (c - o) * 0.5) & ((c - l) > (c - o) * 0.5)

    return bull1 & bull2 & bull3 & higher_close & small_wicks


def is_morning_star(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]

    body1 = (c.shift(2) - o.shift(2))
    big_bearish = body1 < 0
    big_body = body1.abs() > (h.shift(2) - l.shift(2)) * 0.4

    body2 = (c.shift(1) - o.shift(1)).abs()
    range2 = (h.shift(1) - l.shift(1))
    small2 = body2 < range2 * 0.3

    bull3 = c > o
    mid1 = o.shift(2) + body1 / 2
    close_above_mid1 = c > mid1

    prior_down = c.shift(2) < c.shift(3)
    return big_bearish & big_body & small2 & bull3 & close_above_mid1 & prior_down


PATTERN_FUNCS: Dict[str, callable] = {
    "bullish_engulfing": is_bullish_engulfing,
    "hammer": is_hammer,
    "inverted_hammer": is_inverted_hammer,
    "bullish_harami": is_bullish_harami,
    "tweezer_bottom": is_tweezer_bottom,
    "three_white_soldiers": is_three_white_soldiers,
    "morning_star": is_morning_star,
}


# ========= Backtester =========

class PatternBacktester:
    def __init__(self, rr: float = 2.0, max_bars_in_trade: int = 10,
                 use_trend_filter: bool = True, ma_window: int = 50):
        self.rr = rr
        self.max_bars_in_trade = max_bars_in_trade
        self.use_trend_filter = use_trend_filter
        self.ma_window = ma_window

    def _trend_filter_long(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        ma = close.rolling(self.ma_window).mean()
        return close > ma

    def backtest_symbol_pattern(
        self, symbol: str, df: pd.DataFrame, pattern_name: str
    ) -> BacktestStats:
        pattern_func = PATTERN_FUNCS[pattern_name]
        signals = pattern_func(df)

        if self.use_trend_filter:
            trend_ok = self._trend_filter_long(df)
            signals = signals & trend_ok

        trades: List[TradeResult] = []
        in_trade = False
        entry = stop = target = 0.0
        entry_time: Optional[pd.Timestamp] = None
        bars_in_trade = 0

        for i in range(len(df) - 1):
            if not in_trade and bool(signals.iloc[i]):
                next_idx = df.index[i + 1]
                entry = float(df["Open"].iloc[i + 1])

                pattern_low = float(df["Low"].iloc[i])
                stop = pattern_low
                if entry <= stop:
                    continue

                risk_per_share = entry - stop
                target = entry + self.rr * risk_per_share

                entry_time = next_idx
                bars_in_trade = 0
                in_trade = True
                continue

            if in_trade:
                bars_in_trade += 1
                high = float(df["High"].iloc[i])
                low = float(df["Low"].iloc[i])
                close = float(df["Close"].iloc[i])
                cur_time = df.index[i]

                exit_price = None

                if low <= stop:
                    exit_price = stop
                elif high >= target:
                    exit_price = target
                elif bars_in_trade >= self.max_bars_in_trade:
                    exit_price = close

                if exit_price is not None and entry_time is not None:
                    r_multiple = (exit_price - entry) / (entry - stop)
                    trades.append(
                        TradeResult(
                            symbol=symbol,
                            pattern=pattern_name,
                            entry_time=entry_time,
                            exit_time=cur_time,
                            direction="LONG",
                            entry=entry,
                            stop=stop,
                            target=target,
                            exit_price=exit_price,
                            r_multiple=r_multiple,
                            bars_in_trade=bars_in_trade,
                        )
                    )
                    in_trade = False

        return BacktestStats(symbol=symbol, pattern=pattern_name, trades=trades)


# ========= Helpers: summary + export =========

def summarize_stats(all_stats: List[BacktestStats]) -> pd.DataFrame:
    rows = []
    for s in all_stats:
        rows.append(
            {
                "symbol": s.symbol,
                "pattern": s.pattern,
                "num_trades": s.num_trades,
                "win_rate": round(s.win_rate * 100, 2),
                "avg_R": round(s.avg_r, 3),
                "total_R": round(s.total_r, 3),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "pattern", "num_trades", "win_rate", "avg_R", "total_R"]
        )
    return pd.DataFrame(rows)


def export_trades_to_csv(all_stats: List[BacktestStats], filename: str = "trades.csv"):
    rows = []
    for stats in all_stats:
        for t in stats.trades:
            rows.append(
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
    if not rows:
        print("No trades to export.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Exported {len(rows)} trades to {filename}")


def export_to_excel(
    all_stats: List[BacktestStats],
    summary_df: pd.DataFrame,
    filename: str = "backtest_results.xlsx",
):
    # Trades sheet
    trades_rows = []
    for stats in all_stats:
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

    with pd.ExcelWriter(filename) as writer:
        if trades_rows:
            trades_df = pd.DataFrame(trades_rows)
            trades_df.to_excel(writer, sheet_name="trades", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"Exported Excel to {filename}")


# ========= CLI / Runner =========

def run_backtest(
    tickers: List[str],
    interval: str = "1h",
    period: str = "6mo",
    patterns: Optional[List[str]] = None,
) -> List[BacktestStats]:
    if patterns is None or len(patterns) == 0:
        patterns = list(PATTERN_FUNCS.keys())

    backtester = PatternBacktester(
        rr=2.0, max_bars_in_trade=10, use_trend_filter=True, ma_window=50
    )

    all_stats: List[BacktestStats] = []

    for symbol in tickers:
        print(f"\n=== {symbol} | interval={interval} | period={period} ===")
        df = load_ohlc(symbol, interval=interval, period=period)

        for pattern in patterns:
            stats = backtester.backtest_symbol_pattern(symbol, df, pattern)
            all_stats.append(stats)

            print(
                f"Pattern: {pattern:20s} | "
                f"trades={stats.num_trades:3d} | "
                f"win_rate={stats.win_rate*100:5.1f}% | "
                f"avg_R={stats.avg_r:5.2f} | "
                f"total_R={stats.total_r:6.2f}"
            )

    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Candlestick pattern backtester")
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="List of symbols, e.g. --tickers NU AAPL MSFT",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Timeframe/interval, e.g. 1h, 5m, 15m, 1d",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="6mo",
        help="History period, e.g. 6mo, 1mo, 1y",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        help="Optional list of patterns to test; default = all bullish patterns",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="If set, do not export CSV/Excel (only print summary).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_stats = run_backtest(
        tickers=args.tickers,
        interval=args.interval,
        period=args.period,
        patterns=args.patterns,
    )

    summary = summarize_stats(all_stats)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    if not args.no_export:
        export_trades_to_csv(all_stats, filename="trades.csv")
        export_to_excel(all_stats, summary, filename="backtest_results.xlsx")
