"""Numba kernels for Picture Perfect Trading indicators."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be positive")

    out = np.empty(len(values), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.nan

    total = 0.0
    invalid = 0
    for i in range(len(values)):
        value = values[i]
        if np.isfinite(value):
            total += value
        else:
            invalid += 1

        if i >= window:
            old_value = values[i - window]
            if np.isfinite(old_value):
                total -= old_value
            else:
                invalid -= 1

        if i >= window - 1 and invalid == 0:
            out[i] = total / window
    return out


@njit(cache=True)
def _modified_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Wilder-style modified average, seeded by the first full-window SMA."""
    if window < 1:
        raise ValueError("window must be positive")

    out = np.empty(len(values), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.nan

    count = 0
    seed_total = 0.0
    average = np.nan
    for i in range(len(values)):
        value = values[i]
        if not np.isfinite(value):
            count = 0
            seed_total = 0.0
            average = np.nan
            continue

        if count < window:
            count += 1
            seed_total += value
            if count == window:
                average = seed_total / window
                out[i] = average
        else:
            average += (value - average) / window
            out[i] = average
    return out


@njit(cache=True)
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    out = np.empty(len(close), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.nan

    for i in range(len(close)):
        if not np.isfinite(high[i]) or not np.isfinite(low[i]):
            continue

        range_ = high[i] - low[i]
        if i > 0 and np.isfinite(close[i - 1]):
            high_gap = abs(high[i] - close[i - 1])
            low_gap = abs(low[i] - close[i - 1])
            if high_gap > range_:
                range_ = high_gap
            if low_gap > range_:
                range_ = low_gap
        out[i] = range_
    return out


@njit(cache=True)
def commercial_proxy_index(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 8,
) -> np.ndarray:
    """Commercial Proxy: SMA(open-close) / SMA(true range) * 50 + 50."""
    if len(open_) != len(close) or len(high) != len(close) or len(low) != len(close):
        raise ValueError("OHLC arrays must have equal lengths")
    if period < 1:
        raise ValueError("period must be positive")

    open_close = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        open_close[i] = open_[i] - close[i]

    numerator = _rolling_mean(open_close, period)
    denominator = _rolling_mean(_true_range(high, low, close), period)
    out = np.empty(len(close), dtype=np.float64)
    for i in range(len(out)):
        if np.isfinite(numerator[i]) and np.isfinite(denominator[i]) and denominator[i] != 0.0:
            out[i] = numerator[i] / denominator[i] * 50.0 + 50.0
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def large_trader_proxy_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    average_period: int = 40,
    lag: int = 8,
) -> np.ndarray:
    """Large Trader Proxy using Wilder-style modified moving averages."""
    if len(high) != len(close) or len(low) != len(close):
        raise ValueError("OHLC arrays must have equal lengths")
    if average_period < 1:
        raise ValueError("average_period must be positive")
    if lag < 0:
        raise ValueError("lag cannot be negative")

    change = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        change[i] = np.nan
        if i >= lag and np.isfinite(close[i]) and np.isfinite(close[i - lag]):
            change[i] = close[i] - close[i - lag]

    numerator = _modified_moving_average(change, average_period)
    denominator = _modified_moving_average(_true_range(high, low, close), average_period)
    out = np.empty(len(close), dtype=np.float64)
    for i in range(len(out)):
        if np.isfinite(numerator[i]) and np.isfinite(denominator[i]) and denominator[i] != 0.0:
            out[i] = numerator[i] / denominator[i] * 50.0 + 50.0
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def cot_index(commercial_net: np.ndarray, lookback: int) -> np.ndarray:
    """Normalize commercial net positions to their trailing lookback range."""
    if lookback < 1:
        raise ValueError("lookback must be positive")

    out = np.empty(len(commercial_net), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.nan
        if i < lookback - 1:
            continue

        lowest = np.inf
        highest = -np.inf
        valid = True
        for j in range(i - lookback + 1, i + 1):
            value = commercial_net[j]
            if not np.isfinite(value):
                valid = False
                break
            if value < lowest:
                lowest = value
            if value > highest:
                highest = value

        if valid and highest > lowest:
            out[i] = (commercial_net[i] - lowest) / (highest - lowest) * 100.0
    return out


@njit(cache=True)
def _exponential_moving_average(values: np.ndarray, period: int) -> np.ndarray:
    if period < 1:
        raise ValueError("period must be positive")

    out = np.empty(len(values), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.nan

    alpha = 2.0 / (period + 1.0)
    average = np.nan
    for i in range(len(values)):
        value = values[i]
        if not np.isfinite(value):
            continue
        if not np.isfinite(average):
            average = value
        else:
            average += alpha * (value - average)
        out[i] = average
    return out


@njit(cache=True)
def williams_valuation_index(
    market_close: np.ndarray,
    reference_close: np.ndarray,
    fast_period: int = 2,
    slow_period: int = 22,
    normalization_period: int = 156,
) -> np.ndarray:
    """Williams Valuation Index from aligned market/reference close arrays."""
    if len(market_close) != len(reference_close):
        raise ValueError("market and reference arrays must have equal lengths")
    if fast_period < 1 or slow_period < 1 or normalization_period < 1:
        raise ValueError("all periods must be positive")

    spread = np.empty(len(market_close), dtype=np.float64)
    for i in range(len(spread)):
        if (
            np.isfinite(market_close[i])
            and np.isfinite(reference_close[i])
            and reference_close[i] != 0.0
        ):
            spread[i] = market_close[i] / reference_close[i] * 100.0
        else:
            spread[i] = np.nan

    fast = _exponential_moving_average(spread, fast_period)
    slow = _exponential_moving_average(spread, slow_period)
    raw = np.empty(len(spread), dtype=np.float64)
    for i in range(len(raw)):
        if np.isfinite(fast[i]) and np.isfinite(slow[i]):
            raw[i] = slow[i] - fast[i]
        else:
            raw[i] = np.nan

    out = np.empty(len(raw), dtype=np.float64)
    for i in range(len(out)):
        out[i] = np.nan
        if i < normalization_period - 1:
            continue

        lowest = np.inf
        highest = -np.inf
        valid = True
        for j in range(i - normalization_period + 1, i + 1):
            value = raw[j]
            if not np.isfinite(value):
                valid = False
                break
            if value < lowest:
                lowest = value
            if value > highest:
                highest = value

        if valid and highest > lowest:
            out[i] = (raw[i] - lowest) / (highest - lowest) * 100.0
    return out


@njit(cache=True)
def williams_poiv_ad(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    open_interest: np.ndarray,
) -> np.ndarray:
    """Cumulative OI-weighted price change over true range plus On-Balance Volume."""
    size = len(close)
    if len(high) != size or len(low) != size or len(volume) != size or len(open_interest) != size:
        raise ValueError("price, volume, and open-interest arrays must have equal lengths")

    true_range = _true_range(high, low, close)
    out = np.empty(size, dtype=np.float64)
    cumulative_oi_change = 0.0
    on_balance_volume = 0.0
    for i in range(size):
        if i > 0 and np.isfinite(close[i]) and np.isfinite(close[i - 1]):
            price_change = close[i] - close[i - 1]
            if np.isfinite(true_range[i]) and true_range[i] > 0.0 and np.isfinite(open_interest[i]):
                cumulative_oi_change += open_interest[i] * price_change / true_range[i]

            if np.isfinite(volume[i]):
                if price_change > 0.0:
                    on_balance_volume += volume[i]
                elif price_change < 0.0:
                    on_balance_volume -= volume[i]

        out[i] = cumulative_oi_change + on_balance_volume
    return out


@njit(cache=True)
def blast_off_index(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 1,
) -> np.ndarray:
    """Blast Off!: SMA(abs(close-open)) / SMA(true range) * 100."""
    if len(open_) != len(close) or len(high) != len(close) or len(low) != len(close):
        raise ValueError("OHLC arrays must have equal lengths")
    if period < 1:
        raise ValueError("period must be positive")

    open_close = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        open_close[i] = abs(close[i] - open_[i])

    numerator = _rolling_mean(open_close, period)
    denominator = _rolling_mean(_true_range(high, low, close), period)
    out = np.empty(len(close), dtype=np.float64)
    for i in range(len(out)):
        if np.isfinite(numerator[i]) and np.isfinite(denominator[i]) and denominator[i] != 0.0:
            out[i] = numerator[i] / denominator[i] * 100.0
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def willtrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 66,
    multiplier: float = 2.236,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the non-retreating WillTrend line and direction array."""
    if len(high) != len(close) or len(low) != len(close):
        raise ValueError("OHLC arrays must have equal lengths")
    if period < 1:
        raise ValueError("period must be positive")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")

    bar_range = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        if np.isfinite(high[i]) and np.isfinite(low[i]):
            bar_range[i] = high[i] - low[i]
        else:
            bar_range[i] = np.nan
    average_range = _rolling_mean(bar_range, period)

    line_out = np.empty(len(close), dtype=np.float64)
    direction_out = np.empty(len(close), dtype=np.int8)
    for i in range(len(close)):
        line_out[i] = np.nan
        direction_out[i] = 0

    trend = 0
    line = np.nan
    extreme = np.nan
    previous_close = np.nan
    for i in range(len(close)):
        if not np.isfinite(high[i]) or not np.isfinite(low[i]) or not np.isfinite(close[i]):
            trend = 0
            line = np.nan
            extreme = np.nan
            previous_close = np.nan
            continue

        if not np.isfinite(average_range[i]):
            previous_close = close[i]
            continue

        midpoint = (high[i] + low[i]) / 2.0
        if trend == 0:
            if not np.isfinite(previous_close) or close[i] >= previous_close:
                trend = 1
            else:
                trend = -1
            extreme = midpoint

        if trend == 1:
            if midpoint > extreme:
                extreme = midpoint
            candidate = extreme - multiplier * average_range[i]
            if not np.isfinite(line) or candidate > line:
                line = candidate
            if close[i] < line:
                trend = -1
                extreme = midpoint
                line = midpoint + multiplier * average_range[i]
        else:
            if midpoint < extreme:
                extreme = midpoint
            candidate = extreme + multiplier * average_range[i]
            if not np.isfinite(line) or candidate < line:
                line = candidate
            if close[i] > line:
                trend = 1
                extreme = midpoint
                line = midpoint - multiplier * average_range[i]

        line_out[i] = line
        direction_out[i] = trend
        previous_close = close[i]

    return line_out, direction_out
