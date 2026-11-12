# Picture Perfect Indicators

Numba compiled, NumPy array implementations of the indicators and Universal
Entry Technique described in the adjacent *Picture Perfect Trading* manual.
This is a small standalone library; it does not load market data, align
calendars, or simulate orders and position management.

## Included indicators

- **Commercial Proxy Index** — simple moving average of `open - close` divided
  by the simple moving average of true range, scaled by 50 and shifted by 50.
  The manual suggests 8 or 9 bars; the default is 8.
- **Large Trader Proxy Index** — modified moving average of `close - close[lag]`
  divided by the modified moving average of true range, scaled by 50 and shifted
  by 50. The default lag is 8 bars and the default averaging period is 40 bars;
  use an 8 bar averaging period for the manual's weekly setting.
- **COT Index** — trailing min/max normalization of commercial net positions.
  The caller must choose the lookback because the manual discusses multiple
  horizons (including 13, 26, 52, and 156 weeks).
- **Williams Valuation Index** — market/reference close ratio, fast and slow
  exponential averages, their slow-minus-fast difference, then trailing
  min/max normalization. Inputs must already be aligned weekly closes. The
  defaults are 2, 22, and 156 bars.
- **Williams POIV AD** — cumulative open-interest-weighted close change as a
  fraction of true range, added to ordinary On-Balance Volume.
- **Blast Off!** — average absolute open/close change divided by average true
  range, multiplied by 100. The default period is 1.
- **WillTrend** — non-retreating volatility trend line. Its defaults (66 bars
  and a 2.236 multiplier, using the local research toolkit's midpoint and
  high-low average-range interpretation) come from the existing local toolkit;
  the manual describes the method but does not fix those settings.

## Universal Entry Technique

`ringed_swing_points(high, low)` returns confirmed swing high/low prices and
their pivot-bar indices. Swing values are placed on the bar that confirms the
pivot, so they are available without look-ahead. The implementation uses one
bar on either side, strict higher/lower comparisons, and rejects a pivot when
either neighboring bar is an inside bar. An outside bar can mark both a swing
high and low. When consecutive pivots have the same type, the intervening
lowest low or highest high is inserted as the opposite swing.

`universal_entry_technique(open_, high, low, close)` returns four arrays:

- `signal`: `1` for a long entry, `-1` for a short entry, and `0` otherwise.
- `entry_price`: breakout level, or the bar's open when it gaps through the
  trigger; `NaN` when there is no entry.
- `stop_price`: the low of the higher ringed low for a long, or the high of the
  lower ringed high for a short; `NaN` when there is no entry.
- `trend_direction`: inferred direction (`1`, `-1`, or `0` while unresolved).

The function implements the manual's preferred pullback entry: identify a
Williams True Trend Reversal, wait for the first higher ringed low or lower
ringed high, then signal when price breaks that pivot bar's other extreme. It
also implements the described delayed case, where the opposing swing breaks
before a new trend extreme is made and the next pullback must be higher/lower.
The trigger can only activate on the bar after the pivot is confirmed. `close`
is accepted to validate OHLC array lengths; the rules use open/high/low.

Some chart descriptions need explicit computational choices. Trend is seeded
only after both the latest swing high and low are lower than their respective
predecessors (downtrend), or both are higher (uptrend). A true reversal here
requires a fresh extreme in the old trend followed by a break of the last
confirmed opposite swing on a later bar. Breakouts are detected from intrabar
highs/lows. If one bar crosses both a trigger and its stop reference, the
function reports the trigger; OHLC data cannot establish which level traded
first, and the function does not model the subsequent stop fill. These choices
make the output causal and reproducible, but are an interpretation of the
manual's diagrams and prose rather than a verbatim mechanical specification.

The entry function emits signals and the manual's pivot stop reference. It does
not simulate stop fills, exits, position sizing, or the other entry/exit methods
such as WillStop and Target Shooter. The manual also does not provide one
complete computational specification for its seasonal calculation.

## Input and output

Functions accept one-dimensional numeric NumPy arrays and return NumPy arrays.
All kernels are decorated with `@njit(cache=True)`. Inputs to a multi-array
function must have equal lengths. Warm-up values and undefined divisions are
represented by `numpy.nan`. WillTrend returns `(line, direction)`, where
direction is `1` for up, `-1` for down, and `0` during warm-up or missing data.

The functions assume each input series is already on the intended bar
frequency. In particular, the valuation index expects aligned, completed weekly
closes, and COT inputs should be dated by when the report became available if
used in a backtest.

## Interactive presentation

`presentation.qmd` is a RevealJS deck with Plotly charts for each indicator
and a candle chart marking confirmed ringed swing highs and lows with arrows.
The charts are generated from the adjacent local SPY and Gold datasets and
saved under `presentation_assets/`. The Plotly JavaScript library is shared
locally by the chart files.

[Open the interactive presentation](presentation.html) in a browser.

To browse the deck online after pushing, publish the directory containing
`presentation.html` with [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
Keep `presentation_assets/` and `presentation_files/` beside it in the
published files; the deck uses both directories. Add the deployed Pages URL
here once the repository and Pages site are set up.

From the Backtesting workspace root, regenerate the charts and render the deck:

```bash
pixi run --manifest-path 'LarryWilliams/Larry Williams - Picture Perfect Trading/indicators/pixi.toml' figures
quarto render 'LarryWilliams/Larry Williams - Picture Perfect Trading/indicators/presentation.qmd' --to revealjs
```

The POIV AD slide reports input availability because the local OHLCV data does
not include open interest. It does not estimate or substitute that input.

## Run with Pixi

From the Backtesting workspace root:

```bash
PYTHONPATH='LarryWilliams/Larry Williams - Picture Perfect Trading/indicators/src' \
  pixi run --manifest-path 'LarryWilliams/Larry Williams - Picture Perfect Trading/indicators/pixi.toml' \
  python your_script.py
```

Example:

```python
import numpy as np

from picture_perfect_indicators import (
    commercial_proxy_index,
    universal_entry_technique,
)

open_ = np.array([10.0, 10.5, 10.2, 10.8], dtype=np.float64)
high = np.array([10.6, 10.9, 10.7, 11.1], dtype=np.float64)
low = np.array([9.8, 10.1, 10.0, 10.4], dtype=np.float64)
close = np.array([10.3, 10.2, 10.6, 10.9], dtype=np.float64)

index = commercial_proxy_index(open_, high, low, close, period=3)
signals, entries, stops, trend = universal_entry_technique(
    open_, high, low, close
)
```

The formula descriptions above refer to *Picture Perfect Trading* (Larry
Williams, 2007), Sections 2, 3, 5, 6, and 7. The local research implementation
in `../research/` is used only to document the WillTrend settings; this package
does not change that toolkit.
