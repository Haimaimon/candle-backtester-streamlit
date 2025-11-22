#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pattern Validation Layer - 12 Filters + Scoring System
מערכת אימות וציון דפוסים עם 12 פילטרים אמיתיים
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging
import numpy as np
import pandas as pd

# הגדרת logger - מונע דפליקציה
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


@dataclass
class PatternValidationResult:
    """תוצאות אימות הדפוס"""
    is_valid: bool
    pattern_score: float  # 0-100
    volume_before: float
    volume_after: float
    is_trending: bool
    is_support_zone: bool
    rsi_divergence: bool
    candle_range_strength: float
    stop_size_ratio: float
    validation_details: Dict[str, bool]


class PatternValidator:
    """
    מערכת אימות דפוסים עם 12 פילטרים:
    1. Trend Filter (נגד טריידים נגד המגמה)
    2. Support Zone (תמיכה מתחת)
    3. Volume Confirmation (אישור נפח)
    4. Entry Trigger (טריגר כניסה)
    5. Trading Hours (שעות מסחר)
    6. Max Bars Before Pattern (חלון מקסימום)
    7. RSI Divergence (סטיית RSI)
    8. Green Candle Momentum (מומנטום נר ירוק)
    9. Narrow Candles Filter (מניעת נרות צרים)
    10. Pattern Position (מיקום בדפוס)
    11. Stop Size Check (גודל סטופ)
    12. Market Structure (מבנה שוק)
    """

    def __init__(
        self,
        ma_window: int = 50,
        rsi_period: int = 14,
        volume_avg_window: int = 20,
        min_volume_ratio: float = 1.0,  # הורד מ-1.2 ל-1.0 (פחות מחמיר)
        max_bars_before_pattern: int = 20,
        min_candle_range_ratio: float = 0.5,
        max_stop_size_ratio: float = 0.03,  # 3% מקסימום
        trading_start_hour: int = 10,
        trading_end_hour: int = 17,
        enable_logging: bool = True,
    ):
        self.ma_window = ma_window
        self.rsi_period = rsi_period
        self.volume_avg_window = volume_avg_window
        self.min_volume_ratio = min_volume_ratio
        self.max_bars_before_pattern = max_bars_before_pattern
        self.min_candle_range_ratio = min_candle_range_ratio
        self.max_stop_size_ratio = max_stop_size_ratio
        self.trading_start_hour = trading_start_hour
        self.trading_end_hour = trading_end_hour
        self.enable_logging = enable_logging
        self._indicators_cache: Optional[pd.DataFrame] = None  # Cache לאינדיקטורים
        self._support_zones_cache: Optional[pd.Series] = None  # Cache לתמיכות
        
        if self.enable_logging:
            logger.info(f"PatternValidator initialized: MA={ma_window}, RSI={rsi_period}, "
                       f"Volume Window={volume_avg_window}, Min Volume Ratio={min_volume_ratio}")

    def calculate_indicators(self, df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        חישוב כל האינדיקטורים הנדרשים - מיועל עם caching
        """
        # בדיקת cache אם כבר חושבו אינדיקטורים
        if use_cache and self._indicators_cache is not None:
            if len(self._indicators_cache) == len(df):
                if self.enable_logging:
                    logger.debug("Using cached indicators")
                return self._indicators_cache
        
        if self.enable_logging:
            logger.info(f"Calculating indicators for {len(df)} candles...")
        
        df = df.copy()
        
        # MA50 - vectorized
        df['MA50'] = df['Close'].rolling(self.ma_window, min_periods=1).mean()
        
        # RSI - vectorized
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs.replace(np.inf, 100)))
        df['RSI'] = df['RSI'].fillna(50)  # RSI default = 50
        
        # VWAP - vectorized
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumsum_price_volume = (typical_price * df['Volume']).cumsum()
        cumsum_volume = df['Volume'].cumsum()
        df['VWAP'] = cumsum_price_volume / cumsum_volume.replace(0, np.nan)
        df['VWAP'] = df['VWAP'].fillna(df['Close'])  # VWAP default = Close
        
        # Volume Average - vectorized
        df['Volume_Avg'] = df['Volume'].rolling(self.volume_avg_window, min_periods=1).mean()
        
        # Candle body size - vectorized
        df['Body_Size'] = (df['Close'] - df['Open']).abs()
        df['Candle_Range'] = df['High'] - df['Low']
        df['Body_Ratio'] = df['Body_Size'] / df['Candle_Range'].replace(0, np.nan)
        df['Body_Ratio'] = df['Body_Ratio'].fillna(0)
        
        # שמירה ב-cache
        if use_cache:
            self._indicators_cache = df.copy()
        
        if self.enable_logging:
            logger.info("Indicators calculated successfully")
        
        return df

    def find_support_zones(
        self, df: pd.DataFrame, lookback: int = 50, use_cache: bool = True
    ) -> pd.Series:
        """
        מציאת אזורי תמיכה - מיועל עם vectorization ו-caching
        """
        # בדיקת cache
        if use_cache and self._support_zones_cache is not None:
            if len(self._support_zones_cache) == len(df):
                if self.enable_logging:
                    logger.debug("Using cached support zones")
                return self._support_zones_cache
        
        if self.enable_logging:
            logger.info(f"Calculating support zones for {len(df)} candles (lookback={lookback})...")
        
        # חישוב vectorized
        current_lows = df['Low'].values
        vwap_values = df['VWAP'].values if 'VWAP' in df.columns else df['Close'].values
        
        # 1. תמיכה מ-VWAP - vectorized
        vwap_diff = np.abs(current_lows - vwap_values) / np.maximum(vwap_values, 1e-10)
        vwap_support = vwap_diff < 0.005
        
        # 2. תמיכה מה-Low הקודם - vectorized עם rolling
        recent_lows_min = df['Low'].rolling(window=lookback, min_periods=1).min().values
        low_support = np.abs(current_lows - recent_lows_min) / np.maximum(current_lows, 1e-10) < 0.01
        
        # 3. Fibonacci 0.618 - vectorized
        rolling_high = df['High'].rolling(window=lookback, min_periods=1).max().values
        rolling_low = df['Low'].rolling(window=lookback, min_periods=1).min().values
        fib_618 = rolling_low + (rolling_high - rolling_low) * 0.618
        fib_diff = np.abs(current_lows - fib_618) / np.maximum(fib_618, 1e-10)
        fib_support = fib_diff < 0.01
        
        # שילוב כל התנאים
        support_levels = vwap_support | low_support | fib_support
        
        result = pd.Series(support_levels, index=df.index)
        
        # שמירה ב-cache
        if use_cache:
            self._support_zones_cache = result.copy()
        
        if self.enable_logging:
            num_support = result.sum()
            logger.info(f"Support zones calculated: {num_support}/{len(df)} candles have support ({num_support/len(df)*100:.1f}%)")
        
        return result

    def check_market_structure(
        self, df: pd.DataFrame, i: int, lookback: int = 20
    ) -> bool:
        """
        בדיקת מבנה שוק: LH → LL → LH (לפני היפוך)
        Higher High / Lower Low pattern
        """
        if i < lookback * 2:
            return False
        
        start_idx = max(0, i - lookback * 2)
        recent_df = df.iloc[start_idx:i]
        
        # מוצאים נקודות גבוה ונמוך
        highs = recent_df['High'].values
        lows = recent_df['Low'].values
        
        # מחפשים דפוס: ירידה (LL) ואז עלייה (LH)
        # צריך שיהיה Lower Low לפני, ואז נראה עלייה
        if len(recent_df) < 10:
            return False
        
        recent_lows = recent_df['Low'].tail(10).values
        recent_highs = recent_df['High'].tail(10).values
        
        # Lower Low: הנקודה הנמוכה ביותר בסוף
        min_low_idx = np.argmin(recent_lows)
        if min_low_idx == 0:  # הנמוך ביותר הוא האחרון
            return False
        
        # Higher Low אחרי ה-Lower Low
        lows_after_min = recent_lows[min_low_idx+1:]
        if len(lows_after_min) > 0:
            min_low_value = recent_lows[min_low_idx]
            higher_lows = lows_after_min > min_low_value
            return any(higher_lows)
        
        return False

    def validate_pattern(
        self,
        df: pd.DataFrame,
        pattern_idx: int,
        pattern_name: str,
        use_all_filters: bool = True,
        precomputed_indicators: Optional[pd.DataFrame] = None,
        precomputed_support: Optional[pd.Series] = None,
    ) -> PatternValidationResult:
        """
        אימות דפוס עם כל הפילטרים - מיועל
        
        Args:
            df: DataFrame עם נתונים
            pattern_idx: אינדקס של הדפוס
            pattern_name: שם הדפוס
            use_all_filters: האם להשתמש בכל הפילטרים
            precomputed_indicators: אינדיקטורים שכבר חושבו (לאופטימיזציה)
            precomputed_support: תמיכות שכבר חושבו (לאופטימיזציה)
        
        Returns:
            PatternValidationResult עם כל הפרטים
        """
        if self.enable_logging:
            logger.debug(f"Validating pattern {pattern_name} at index {pattern_idx}")
        
        if pattern_idx < self.ma_window or pattern_idx >= len(df) - 1:
            if self.enable_logging:
                logger.debug(f"Pattern {pattern_name} at {pattern_idx}: Invalid index (idx < MA window or >= len-1)")
            return PatternValidationResult(
                is_valid=False,
                pattern_score=0.0,
                volume_before=0.0,
                volume_after=0.0,
                is_trending=False,
                is_support_zone=False,
                rsi_divergence=False,
                candle_range_strength=0.0,
                stop_size_ratio=0.0,
                validation_details={}
            )
        
        # שימוש ב-precomputed indicators אם קיימים, אחרת חישוב
        if precomputed_indicators is not None:
            df = precomputed_indicators
        else:
            df = self.calculate_indicators(df, use_cache=True)
        
        validation_details = {}
        
        # ========= Filter 1: Trend Filter =========
        close = df['Close'].iloc[pattern_idx]
        ma50 = df['MA50'].iloc[pattern_idx]
        is_trending = close > ma50
        validation_details['trend_filter'] = is_trending
        
        if self.enable_logging:
            logger.debug(f"Filter 1 (Trend): Close={close:.4f}, MA50={ma50:.4f}, is_trending={is_trending}")
        
        # ========= Filter 2: Support Zone =========
        if precomputed_support is not None:
            support_zones = precomputed_support
        else:
            support_zones = self.find_support_zones(df, use_cache=True)
        is_support_zone = support_zones.iloc[pattern_idx] if pattern_idx < len(support_zones) else False
        validation_details['support_zone'] = is_support_zone
        
        if self.enable_logging:
            logger.debug(f"Filter 2 (Support Zone): {is_support_zone}")
        
        # ========= Filter 3: Volume Confirmation =========
        volume_before = df['Volume'].iloc[pattern_idx - 1] if pattern_idx > 0 else 0
        volume_after = df['Volume'].iloc[pattern_idx]
        volume_avg = df['Volume_Avg'].iloc[pattern_idx]
        
        # הקלה משמעותית על volume confirmation
        # דרישות: ווליום גבוה מהממוצע (לא חובה שהוא גבוה יותר מהווליום הקודם)
        volume_confirmation = volume_after > volume_avg * self.min_volume_ratio
        validation_details['volume_confirmation'] = volume_confirmation
        
        if self.enable_logging:
            logger.debug(f"Filter 3 (Volume): before={volume_before:.0f}, after={volume_after:.0f}, "
                        f"avg={volume_avg:.0f}, confirmed={volume_confirmation}")
        
        # ========= Filter 4: Entry Trigger =========
        # בדיקה: האם הנר הבא עובר את ה-High של הדפוס
        # הערה: זה ייבדק שוב ב-Backtester לפני הכניסה
        if pattern_idx < len(df) - 1:
            pattern_high = df['High'].iloc[pattern_idx]
            next_high = df['High'].iloc[pattern_idx + 1]
            entry_trigger = next_high > pattern_high
        else:
            entry_trigger = False  # אין נר הבא, לא ניתן לבדוק
        validation_details['entry_trigger'] = entry_trigger
        
        # ========= Filter 5: Trading Hours =========
        # הפוך לאופציונלי יותר - לא חובה בשביל validation
        pattern_time = df.index[pattern_idx]
        if hasattr(pattern_time, 'hour'):
            hour = pattern_time.hour
            trading_hours_ok = self.trading_start_hour <= hour < self.trading_end_hour
        else:
            trading_hours_ok = True  # אם אין זמן, מניחים שזה בסדר
        validation_details['trading_hours'] = trading_hours_ok
        # הערה: trading_hours לא ייכלל ב-is_valid (לא חובה)
        
        # ========= Filter 6: Max Bars Before Pattern =========
        # הקלה משמעותית - הורד את הסף מ-0.005 ל-0.002
        if pattern_idx >= self.max_bars_before_pattern:
            bars_before = df.iloc[pattern_idx - self.max_bars_before_pattern:pattern_idx]
            price_volatility = bars_before['Close'].std() / bars_before['Close'].mean()
            max_bars_ok = price_volatility > 0.002  # הורד מ-0.005 ל-0.002 (פחות מחמיר)
        else:
            max_bars_ok = True
        validation_details['max_bars_ok'] = max_bars_ok
        # הערה: max_bars_ok לא ייכלל ב-is_valid (לא חובה)
        
        # ========= Filter 7: RSI Divergence =========
        rsi_current = df['RSI'].iloc[pattern_idx]
        rsi_prev = df['RSI'].iloc[pattern_idx - 1] if pattern_idx > 0 else rsi_current
        low_current = df['Low'].iloc[pattern_idx]
        low_prev = df['Low'].iloc[pattern_idx - 1] if pattern_idx > 0 else low_current
        
        rsi_divergence = (rsi_current > rsi_prev) and (low_current < low_prev)
        validation_details['rsi_divergence'] = rsi_divergence
        
        # ========= Filter 8: Green Candle Momentum =========
        green_body = df['Close'].iloc[pattern_idx] > df['Open'].iloc[pattern_idx]
        body_ratio = df['Body_Ratio'].iloc[pattern_idx] if 'Body_Ratio' in df.columns else 0
        green_momentum = green_body and (body_ratio > 0.5)
        validation_details['green_momentum'] = green_momentum
        
        # ========= Filter 9: Narrow Candles Filter =========
        # הקלה - הורד את הסף מ-0.5 ל-0.3
        candle_range = df['Candle_Range'].iloc[pattern_idx]
        avg_range = df['Candle_Range'].rolling(20).mean().iloc[pattern_idx]
        not_narrow = candle_range >= avg_range * 0.3  # הורד מ-0.5 ל-0.3 (פחות מחמיר)
        validation_details['not_narrow'] = not_narrow
        
        # ========= Filter 10: Pattern Position (bottom of movement) =========
        # הקלה - הורד את הסף מ-1% ל-2%
        if pattern_idx >= 20:
            recent_lows = df['Low'].iloc[pattern_idx - 20:pattern_idx + 1]
            current_low = df['Low'].iloc[pattern_idx]
            is_at_bottom = current_low <= recent_lows.min() * 1.02  # הורד מ-1% ל-2% (פחות מחמיר)
        else:
            is_at_bottom = True  # לא מספיק היסטוריה
        validation_details['is_at_bottom'] = is_at_bottom
        # הערה: is_at_bottom לא ייכלל ב-is_valid (לא חובה)
        
        # ========= Filter 11: Stop Size Check =========
        entry_price = df['Open'].iloc[pattern_idx + 1] if pattern_idx < len(df) - 1 else df['Close'].iloc[pattern_idx]
        stop_price = df['Low'].iloc[pattern_idx]
        if entry_price > 0:
            stop_size_ratio = (entry_price - stop_price) / entry_price
            stop_size_ok = stop_size_ratio <= self.max_stop_size_ratio
        else:
            stop_size_ratio = 0.0
            stop_size_ok = False
        validation_details['stop_size_ok'] = stop_size_ok
        
        # ========= Filter 12: Market Structure =========
        market_structure_ok = self.check_market_structure(df, pattern_idx)
        validation_details['market_structure'] = market_structure_ok
        
        # ========= Pattern Scoring (0-100) =========
        score_components = {
            'low_match': 20.0,  # ייבדק לפי דפוס ספציפי
            'green_reversal': 15.0,
            'volume_spike': 20.0,
            'downtrend_before': 10.0,
            'support_zone': 20.0,
            'big_green_body': 10.0,
            'rsi_divergence': 5.0,
        }
        
        pattern_score = 0.0
        
        # Low match (תלוי בדפוס)
        if pattern_name == 'tweezer_bottom':
            if pattern_idx > 0:
                low_match = abs(df['Low'].iloc[pattern_idx] - df['Low'].iloc[pattern_idx - 1]) / df['Low'].iloc[pattern_idx] < 0.005
                if low_match:
                    pattern_score += score_components['low_match']
        
        # Green reversal
        if green_body and df['Close'].iloc[pattern_idx] > df['Open'].iloc[pattern_idx]:
            pattern_score += score_components['green_reversal']
        
        # Volume spike
        if volume_confirmation:
            pattern_score += score_components['volume_spike']
        
        # Downtrend before
        if pattern_idx > 5:
            prev_close = df['Close'].iloc[pattern_idx - 5]
            current_close = df['Close'].iloc[pattern_idx]
            if current_close < prev_close * 0.98:
                pattern_score += score_components['downtrend_before']
        
        # Support zone
        if is_support_zone:
            pattern_score += score_components['support_zone']
        
        # Big green body
        if green_momentum:
            pattern_score += score_components['big_green_body']
        
        # RSI divergence
        if rsi_divergence:
            pattern_score += score_components['rsi_divergence']
        
        # חישוב candle range strength
        candle_range_strength = min(1.0, candle_range / avg_range) if avg_range > 0 else 0.0
        
        # ========= Final Validation =========
        if use_all_filters:
            # Entry Trigger נבדק בנפרד ב-Backtester לפני הכניסה
            # אז לא נכלול אותו כאן ב-is_valid
            
            # לוגיקה פחות מחמירה: רוב הפילטרים צריכים לעבור, לא כולם
            # פילטרים קריטיים (חייבים): trend, volume, stop_size, not_narrow
            # פילטרים אופציונליים: trading_hours, max_bars_ok, is_at_bottom
            
            critical_filters = [
                is_trending,
                volume_confirmation,
                stop_size_ok,
                not_narrow,
            ]
            
            optional_filters = [
                trading_hours_ok,
                max_bars_ok,
                is_at_bottom,
            ]
            
            # דורשים שכל הפילטרים הקריטיים יעברו + לפחות 2 מתוך 3 האופציונליים
            critical_passed = all(critical_filters)
            optional_passed_count = sum(optional_filters)
            
            # אפשר גם להיות יותר גמישים - רק רוב הפילטרים הקריטיים
            # critical_passed = sum(critical_filters) >= 3  # לפחות 3 מתוך 4
            
            is_valid = critical_passed and (optional_passed_count >= 1)  # לפחות פילטר אופציונלי אחד
        else:
            # אם לא משתמשים בכל הפילטרים, רק ציון מינימלי
            is_valid = pattern_score >= 40.0
        
        if self.enable_logging:
            passed_filters = sum(validation_details.values())
            total_filters = len(validation_details)
            failed_filters_list = [name for name, passed in validation_details.items() if not passed]
            
            # לוג מפורט רק אם הדפוס לא עבר (כדי לא להציף לוגים)
            if not is_valid and len(failed_filters_list) > 0:
                # לוג רק במקרים מעטים (לא להציף)
                if len(failed_filters_list) <= 5:  # רק אם יש מעט פילטרים שנכשלו
                    logger.debug(
                        f"Pattern {pattern_name} at {pattern_idx}: "
                        f"Valid={is_valid}, Score={pattern_score:.1f}/100, "
                        f"Filters passed: {passed_filters}/{total_filters}, "
                        f"Failed: {', '.join(failed_filters_list[:3])}"
                    )
            # לוג רק לדפוסים שעברו (חשובים יותר)
            elif is_valid:
                logger.info(
                    f"Pattern {pattern_name} at {pattern_idx}: "
                    f"✅ Valid={is_valid}, Score={pattern_score:.1f}/100, "
                    f"Filters passed: {passed_filters}/{total_filters}"
                )
        
        return PatternValidationResult(
            is_valid=is_valid,
            pattern_score=pattern_score,
            volume_before=volume_before,
            volume_after=volume_after,
            is_trending=is_trending,
            is_support_zone=is_support_zone,
            rsi_divergence=rsi_divergence,
            candle_range_strength=candle_range_strength,
            stop_size_ratio=stop_size_ratio,
            validation_details=validation_details
        )
    
    def reset_cache(self):
        """איפוס cache - שימושי כשמשנים DataFrame"""
        self._indicators_cache = None
        self._support_zones_cache = None
        if self.enable_logging:
            logger.info("Cache reset")


def score_pattern(
    df: pd.DataFrame,
    pattern_idx: int,
    pattern_name: str,
    validator: Optional[PatternValidator] = None,
) -> float:
    """
    פונקציית ציון פשוטה - מחזירה ציון 0-100
    
    Args:
        df: DataFrame עם נתונים
        pattern_idx: אינדקס של הדפוס
        pattern_name: שם הדפוס
        validator: אובייקט PatternValidator (אם None, יוצר חדש)
    
    Returns:
        ציון בין 0 ל-100
    """
    if validator is None:
        validator = PatternValidator()
    
    result = validator.validate_pattern(df, pattern_idx, pattern_name, use_all_filters=False)
    return result.pattern_score

