"""Numba-compatible entry methods from Picture Perfect Trading."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _is_inside_bar(high: np.ndarray, low: np.ndarray, index: int) -> bool:
    if index < 1:
        return False
    return (
        np.isfinite(high[index])
        and np.isfinite(low[index])
        and np.isfinite(high[index - 1])
        and np.isfinite(low[index - 1])
        and high[index] < high[index - 1]
        and low[index] > low[index - 1]
    )


@njit(cache=True)
def ringed_swing_points(
    high: np.ndarray, low: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return confirmed ringed swing prices and their pivot-bar indices.

    Swing events are placed on the bar when they become knowable (the bar to
    the right of the pivot has closed). Values are NaN and indices are -1 when
    there is no event. If consecutive swings have the same type, the intervening
    extreme is inserted as the opposite swing, following the manual's rule.
    """
    if len(high) != len(low):
        raise ValueError("high and low arrays must have equal lengths")

    size = len(high)
    swing_high = np.empty(size, dtype=np.float64)
    swing_low = np.empty(size, dtype=np.float64)
    swing_high_index = np.empty(size, dtype=np.int64)
    swing_low_index = np.empty(size, dtype=np.int64)
    for i in range(size):
        swing_high[i] = np.nan
        swing_low[i] = np.nan
        swing_high_index[i] = -1
        swing_low_index[i] = -1

    last_high_index = -1
    last_low_index = -1
    for confirm_index in range(2, size):
        pivot_index = confirm_index - 1
        if (
            not np.isfinite(high[pivot_index - 1])
            or not np.isfinite(high[pivot_index])
            or not np.isfinite(high[pivot_index + 1])
            or not np.isfinite(low[pivot_index - 1])
            or not np.isfinite(low[pivot_index])
            or not np.isfinite(low[pivot_index + 1])
        ):
            continue

        is_swing_high = (
            high[pivot_index] > high[pivot_index - 1]
            and high[pivot_index] > high[pivot_index + 1]
        )
        is_swing_low = (
            low[pivot_index] < low[pivot_index - 1]
            and low[pivot_index] < low[pivot_index + 1]
        )
        if not is_swing_high and not is_swing_low:
            continue

        if _is_inside_bar(high, low, pivot_index - 1) or _is_inside_bar(
            high, low, pivot_index + 1
        ):
            continue

        if is_swing_high and is_swing_low:
            swing_high[confirm_index] = high[pivot_index]
            swing_high_index[confirm_index] = pivot_index
            swing_low[confirm_index] = low[pivot_index]
            swing_low_index[confirm_index] = pivot_index
            last_high_index = pivot_index
            last_low_index = pivot_index
            continue

        if is_swing_high:
            if last_high_index >= 0 and last_low_index <= last_high_index:
                lowest_index = -1
                lowest_value = np.inf
                for j in range(last_high_index + 1, pivot_index):
                    if np.isfinite(low[j]) and low[j] < lowest_value:
                        lowest_value = low[j]
                        lowest_index = j
                if lowest_index >= 0:
                    swing_low[confirm_index] = lowest_value
                    swing_low_index[confirm_index] = lowest_index
                    last_low_index = lowest_index

            swing_high[confirm_index] = high[pivot_index]
            swing_high_index[confirm_index] = pivot_index
            last_high_index = pivot_index

        if is_swing_low:
            if last_low_index >= 0 and last_high_index <= last_low_index:
                highest_index = -1
                highest_value = -np.inf
                for j in range(last_low_index + 1, pivot_index):
                    if np.isfinite(high[j]) and high[j] > highest_value:
                        highest_value = high[j]
                        highest_index = j
                if highest_index >= 0:
                    swing_high[confirm_index] = highest_value
                    swing_high_index[confirm_index] = highest_index
                    last_high_index = highest_index

            swing_low[confirm_index] = low[pivot_index]
            swing_low_index[confirm_index] = pivot_index
            last_low_index = pivot_index

    return swing_high, swing_low, swing_high_index, swing_low_index


@njit(cache=True)
def universal_entry_technique(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Universal Entry Technique signals from OHLC arrays.

    Returns ``(signal, entry_price, stop_price, trend_direction)``. Signals are
    1 for long, -1 for short, and 0 otherwise. Entry prices are the breakout
    level, or the opening price when a bar gaps through it. Stops are the low
    of the higher ringed low for longs and the high of the lower ringed high
    for shorts. Trend direction is 1, -1, or 0 while structure is unresolved.

    The preferred post-reversal pullback entry is implemented. A Williams True
    Trend Reversal needs a fresh extreme in the existing trend followed by a
    break of the prior opposite swing. The manual's delayed case is also
    included: after a swing point forms and the opposite swing is broken, wait
    for the next higher low/lower high before arming the entry.

    Ringed points use one bar on each side and strict comparisons. A neighboring
    inside bar disqualifies a point. Breakouts use intrabar highs/lows. Since a
    pivot is only confirmed at the close of its right-hand bar, its breakout
    trigger can be acted on starting with the following bar.
    """
    size = len(close)
    if len(open_) != size or len(high) != size or len(low) != size:
        raise ValueError("OHLC arrays must have equal lengths")

    signal = np.zeros(size, dtype=np.int8)
    entry_price = np.empty(size, dtype=np.float64)
    stop_price = np.empty(size, dtype=np.float64)
    trend_direction = np.zeros(size, dtype=np.int8)
    for i in range(size):
        entry_price[i] = np.nan
        stop_price[i] = np.nan

    swing_high, swing_low, swing_high_index, swing_low_index = ringed_swing_points(
        high, low
    )

    previous_swing_high = np.nan
    previous_swing_low = np.nan
    last_swing_high = np.nan
    last_swing_low = np.nan
    last_swing_high_index = -1
    last_swing_low_index = -1

    trend = 0
    trend_extreme = np.nan
    trend_extreme_bar = -1
    extended_in_trend = False

    long_setup_kind = 0  # 1 = Williams reversal, 2 = delayed reversal
    long_reference = np.nan
    long_reversal_bar = -1
    long_trigger = np.nan
    long_stop = np.nan
    long_available_from = -1

    short_setup_kind = 0
    short_reference = np.nan
    short_reversal_bar = -1
    short_trigger = np.nan
    short_stop = np.nan
    short_available_from = -1

    for i in range(size):
        prior_high = last_swing_high
        prior_low = last_swing_low
        prior_high_index = last_swing_high_index
        prior_low_index = last_swing_low_index

        new_swing_high = np.isfinite(swing_high[i])
        new_swing_low = np.isfinite(swing_low[i])

        event_count = int(new_swing_high) + int(new_swing_low)
        first_event_is_high = new_swing_high
        if new_swing_high and new_swing_low:
            first_event_is_high = swing_high_index[i] <= swing_low_index[i]

        for event_slot in range(2):
            if event_slot >= event_count:
                break
            if event_slot == 0:
                event_is_high = first_event_is_high
            else:
                event_is_high = not first_event_is_high

            if event_is_high:
                if short_setup_kind != 0:
                    pivot_index = swing_high_index[i]
                    if pivot_index > short_reversal_bar:
                        if swing_high[i] < short_reference:
                            short_trigger = low[pivot_index]
                            short_stop = high[pivot_index]
                            short_available_from = i + 1
                            short_setup_kind = 0
                        else:
                            if short_setup_kind == 1:
                                trend = 1
                                trend_extreme = swing_high[i]
                                trend_extreme_bar = swing_high_index[i]
                                extended_in_trend = False
                            short_setup_kind = 0

                previous_swing_high = last_swing_high
                last_swing_high = swing_high[i]
                last_swing_high_index = swing_high_index[i]
            else:
                if long_setup_kind != 0:
                    pivot_index = swing_low_index[i]
                    if pivot_index > long_reversal_bar:
                        if swing_low[i] > long_reference:
                            long_trigger = high[pivot_index]
                            long_stop = low[pivot_index]
                            long_available_from = i + 1
                            long_setup_kind = 0
                        else:
                            if long_setup_kind == 1:
                                trend = -1
                                trend_extreme = swing_low[i]
                                trend_extreme_bar = swing_low_index[i]
                                extended_in_trend = False
                            long_setup_kind = 0

                previous_swing_low = last_swing_low
                last_swing_low = swing_low[i]
                last_swing_low_index = swing_low_index[i]

        initialized_now = False
        if trend == 0 and np.isfinite(previous_swing_high) and np.isfinite(
            previous_swing_low
        ):
            if (
                last_swing_high < previous_swing_high
                and last_swing_low < previous_swing_low
            ):
                trend = -1
                trend_extreme = last_swing_low
                trend_extreme_bar = last_swing_low_index
                extended_in_trend = False
                initialized_now = True
            elif (
                last_swing_high > previous_swing_high
                and last_swing_low > previous_swing_low
            ):
                trend = 1
                trend_extreme = last_swing_high
                trend_extreme_bar = last_swing_high_index
                extended_in_trend = False
                initialized_now = True

        reversed_now = False
        if trend == -1 and not initialized_now:
            if np.isfinite(low[i]) and (
                not np.isfinite(trend_extreme) or low[i] < trend_extreme
            ):
                trend_extreme = low[i]
                trend_extreme_bar = i
                extended_in_trend = True

            if (
                np.isfinite(prior_high)
                and np.isfinite(high[i])
                and high[i] > prior_high
            ):
                if extended_in_trend and i > trend_extreme_bar:
                    trend = 1
                    long_reference = trend_extreme
                    trend_extreme = high[i]
                    trend_extreme_bar = i
                    extended_in_trend = False
                    long_setup_kind = 1
                    long_reversal_bar = i
                    long_trigger = np.nan
                    long_stop = np.nan
                    short_setup_kind = 0
                    short_trigger = np.nan
                    short_stop = np.nan
                    reversed_now = True
                elif (
                    not extended_in_trend
                    and prior_low_index > prior_high_index
                    and np.isfinite(prior_low)
                ):
                    long_setup_kind = 2
                    long_reference = prior_low
                    long_reversal_bar = i
                    long_trigger = np.nan
                    long_stop = np.nan
                    short_setup_kind = 0
                    short_trigger = np.nan
                    short_stop = np.nan

        elif trend == 1 and not initialized_now:
            if np.isfinite(high[i]) and (
                not np.isfinite(trend_extreme) or high[i] > trend_extreme
            ):
                trend_extreme = high[i]
                trend_extreme_bar = i
                extended_in_trend = True

            if np.isfinite(prior_low) and np.isfinite(low[i]) and low[i] < prior_low:
                if extended_in_trend and i > trend_extreme_bar:
                    trend = -1
                    short_reference = trend_extreme
                    trend_extreme = low[i]
                    trend_extreme_bar = i
                    extended_in_trend = False
                    short_setup_kind = 1
                    short_reversal_bar = i
                    short_trigger = np.nan
                    short_stop = np.nan
                    long_setup_kind = 0
                    long_trigger = np.nan
                    long_stop = np.nan
                    reversed_now = True
                elif (
                    not extended_in_trend
                    and prior_high_index > prior_low_index
                    and np.isfinite(prior_high)
                ):
                    short_setup_kind = 2
                    short_reference = prior_high
                    short_reversal_bar = i
                    short_trigger = np.nan
                    short_stop = np.nan
                    long_setup_kind = 0
                    long_trigger = np.nan
                    long_stop = np.nan

        if not reversed_now:
            if np.isfinite(long_trigger) and i >= long_available_from:
                if np.isfinite(high[i]) and high[i] > long_trigger:
                    signal[i] = 1
                    if np.isfinite(open_[i]) and open_[i] > long_trigger:
                        entry_price[i] = open_[i]
                    else:
                        entry_price[i] = long_trigger
                    stop_price[i] = long_stop
                    trend = 1
                    trend_extreme = high[i]
                    trend_extreme_bar = i
                    extended_in_trend = False
                    long_trigger = np.nan
                    long_stop = np.nan
                    short_trigger = np.nan
                    short_stop = np.nan
                elif np.isfinite(low[i]) and low[i] <= long_stop:
                    long_trigger = np.nan
                    long_stop = np.nan

            if (
                signal[i] == 0
                and np.isfinite(short_trigger)
                and i >= short_available_from
            ):
                if np.isfinite(low[i]) and low[i] < short_trigger:
                    signal[i] = -1
                    if np.isfinite(open_[i]) and open_[i] < short_trigger:
                        entry_price[i] = open_[i]
                    else:
                        entry_price[i] = short_trigger
                    stop_price[i] = short_stop
                    trend = -1
                    trend_extreme = low[i]
                    trend_extreme_bar = i
                    extended_in_trend = False
                    short_trigger = np.nan
                    short_stop = np.nan
                    long_trigger = np.nan
                    long_stop = np.nan
                elif np.isfinite(high[i]) and high[i] >= short_stop:
                    short_trigger = np.nan
                    short_stop = np.nan

        trend_direction[i] = trend

    return signal, entry_price, stop_price, trend_direction
