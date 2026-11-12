"""Build the Plotly HTML charts embedded in presentation.qmd."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from picture_perfect_indicators import (  # noqa: E402
    blast_off_index,
    commercial_proxy_index,
    cot_index,
    large_trader_proxy_index,
    ringed_swing_points,
    williams_poiv_ad,
    williams_valuation_index,
    willtrend,
)

SPY_PATH = HERE.parent / "research/data/spy-features.parquet"
GOLD_PATH = HERE.parent / "research/data/fortress/xauusd_1d.parquet"
GOLD_COT_PATH = HERE.parent / "research/data/fortress/gold-cot.parquet"
OUTPUT_DIR = HERE / "presentation_assets"
DISPLAY_FROM = pd.Timestamp("2023-01-01")


def style_figure(fig: go.Figure, title: str, height: int = 540) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#10151d",
        plot_bgcolor="#10151d",
        font={"color": "#e8edf4", "size": 13},
        width=1120,
        height=height,
        margin={"l": 55, "r": 35, "t": 60, "b": 40},
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    return fig


def write_figure(
    fig: go.Figure, filename: str, title: str, height: int = 540
) -> None:
    style_figure(fig, title, height=height)
    fig.write_html(
        OUTPUT_DIR / filename,
        full_html=True,
        include_plotlyjs="directory",
        config={"responsive": True, "displaylogo": False},
    )


def add_candles(
    fig: go.Figure,
    dates: pd.Series | pd.Index,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    name: str,
    row: int,
    col: int,
    secondary_y: bool | None = None,
) -> None:
    trace = go.Candlestick(
        x=dates,
        open=open_,
        high=high,
        low=low,
        close=close,
        name=name,
        increasing_line_color="#62d6a5",
        increasing_fillcolor="#237b62",
        decreasing_line_color="#ff6b6b",
        decreasing_fillcolor="#8f3540",
        whiskerwidth=0.4,
    )
    if secondary_y is None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)


def add_ringed_swing_arrows(
    fig: go.Figure,
    dates: pd.Series,
    high: np.ndarray,
    low: np.ndarray,
    visible: np.ndarray,
) -> tuple[int, int]:
    """Annotate confirmed ringed swing highs/lows on a candlestick chart."""
    _, _, swing_high_indices, swing_low_indices = ringed_swing_points(high, low)
    high_pivots = swing_high_indices[swing_high_indices >= 0]
    low_pivots = swing_low_indices[swing_low_indices >= 0]
    high_pivots = high_pivots[visible[high_pivots]]
    low_pivots = low_pivots[visible[low_pivots]]

    for pivot in high_pivots:
        fig.add_annotation(
            x=dates.iloc[pivot],
            y=high[pivot],
            xref="x",
            yref="y",
            ax=0,
            ay=-14,
            axref="pixel",
            ayref="pixel",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1.2,
            arrowcolor="#ff8f70",
        )

    for pivot in low_pivots:
        fig.add_annotation(
            x=dates.iloc[pivot],
            y=low[pivot],
            xref="x",
            yref="y",
            ax=0,
            ay=14,
            axref="pixel",
            ayref="pixel",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1.2,
            arrowcolor="#62d6a5",
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Ringed swing high",
            marker={"symbol": "triangle-down", "size": 8, "color": "#ff8f70"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Ringed swing low",
            marker={"symbol": "triangle-up", "size": 8, "color": "#62d6a5"},
        ),
        row=1,
        col=1,
    )
    return len(high_pivots), len(low_pivots)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spy = pd.read_parquet(SPY_PATH)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = (
        spy.sort_values("date")
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index(drop=True)
    )

    gold = pd.read_parquet(GOLD_PATH)
    gold["date"] = pd.to_datetime(gold["date"], utc=True).dt.tz_convert(None)
    gold = gold.sort_values("date").dropna(subset=["close"])

    gold_cot = pd.read_parquet(GOLD_COT_PATH, columns=["date", "commercial_net"])
    gold_cot["date"] = pd.to_datetime(gold_cot["date"], utc=True).dt.tz_convert(None)
    gold_cot = gold_cot.sort_values("date").reset_index(drop=True)
    return spy, gold, gold_cot


def price_indicator_figure(
    dates: pd.Series,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    reference: float | None = None,
) -> go.Figure:
    visible = (dates >= DISPLAY_FROM).to_numpy()
    chart_dates = dates[visible]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
        subplot_titles=("SPY daily OHLC candles", label),
    )
    add_candles(
        fig,
        chart_dates,
        open_[visible],
        high[visible],
        low[visible],
        close[visible],
        name="SPY OHLC",
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_dates,
            y=values[visible],
            mode="lines",
            name=label,
            line={"color": color, "width": 1.8},
        ),
        row=2,
        col=1,
    )
    if reference is not None:
        fig.add_hline(y=reference, line_dash="dot", line_color="#8793a3", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Index", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    return fig


def build_charts() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spy, gold, gold_cot = load_inputs()

    open_ = spy["open"].to_numpy(dtype=np.float64)
    high = spy["high"].to_numpy(dtype=np.float64)
    low = spy["low"].to_numpy(dtype=np.float64)
    close = spy["close"].to_numpy(dtype=np.float64)
    volume = spy["volume"].to_numpy(dtype=np.float64)
    dates = spy["date"]

    commercial = commercial_proxy_index(open_, high, low, close, period=8)
    write_figure(
        price_indicator_figure(
            dates, open_, high, low, close, commercial, "Commercial Proxy · 8 bars", "#62d6a5", 50
        ),
        "commercial-proxy.html",
        "Commercial Proxy Index · SPY daily",
    )

    large_trader = large_trader_proxy_index(high, low, close, average_period=40, lag=8)
    write_figure(
        price_indicator_figure(
            dates, open_, high, low, close, large_trader,
            "Large Trader Proxy · 40 bar average", "#f6bd60", 50
        ),
        "large-trader-proxy.html",
        "Large Trader Proxy Index · SPY daily",
    )

    gold_cot["cot_index"] = cot_index(
        gold_cot["commercial_net"].to_numpy(dtype=np.float64), lookback=13
    )
    cot_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=("Gold daily OHLC candles", "Gold commercial positioning and COT Index"),
    )
    add_candles(
        cot_figure,
        gold["date"],
        gold["open"].to_numpy(dtype=np.float64),
        gold["high"].to_numpy(dtype=np.float64),
        gold["low"].to_numpy(dtype=np.float64),
        gold["close"].to_numpy(dtype=np.float64),
        name="Gold OHLC",
        row=1,
        col=1,
    )
    cot_figure.add_trace(
        go.Scatter(
            x=gold_cot["date"],
            y=gold_cot["cot_index"],
            mode="lines",
            name="13 report COT Index",
            line={"color": "#bd93f9", "width": 2},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    cot_figure.add_trace(
        go.Scatter(
            x=gold_cot["date"],
            y=gold_cot["commercial_net"],
            mode="lines",
            name="Commercial net positions",
            line={"color": "#57c7ff", "width": 1.2},
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    for level in (25, 75):
        cot_figure.add_hline(
            y=level,
            line_dash="dot",
            line_color="#8793a3",
            row=2,
            col=1,
            secondary_y=False,
        )
    cot_figure.update_yaxes(title_text="Gold price", row=1, col=1)
    cot_figure.update_yaxes(
        title_text="COT Index (0–100)", range=[0, 100], row=2, col=1, secondary_y=False
    )
    cot_figure.update_yaxes(
        title_text="Commercial net", row=2, col=1, secondary_y=True
    )
    cot_figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    write_figure(cot_figure, "cot-index.html", "Gold COT Index · weekly reports")

    spy_weekly = (
        spy.set_index("date")[["open", "high", "low", "close"]]
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    gold_weekly = (
        gold.set_index("date")["close"].resample("W-FRI").last().dropna().rename("gold")
    )
    valuation_pair = spy_weekly.join(gold_weekly, how="inner").dropna()
    valuation = williams_valuation_index(
        valuation_pair["close"].to_numpy(dtype=np.float64),
        valuation_pair["gold"].to_numpy(dtype=np.float64),
        fast_period=2,
        slow_period=22,
        normalization_period=156,
    )
    valuation_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=("Weekly SPY candles and SPY / Gold ratio", "Williams Valuation Index"),
    )
    add_candles(
        valuation_figure,
        valuation_pair.index,
        valuation_pair["open"].to_numpy(dtype=np.float64),
        valuation_pair["high"].to_numpy(dtype=np.float64),
        valuation_pair["low"].to_numpy(dtype=np.float64),
        valuation_pair["close"].to_numpy(dtype=np.float64),
        name="SPY weekly OHLC",
        row=1,
        col=1,
        secondary_y=False,
    )
    valuation_figure.add_trace(
        go.Scatter(
            x=valuation_pair.index,
            y=valuation_pair["close"] / valuation_pair["gold"] * 100,
            mode="lines",
            name="SPY / Gold × 100",
            line={"color": "#57c7ff", "width": 1.6},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    valuation_figure.add_trace(
        go.Scatter(
            x=valuation_pair.index,
            y=valuation,
            mode="lines",
            name="WVI",
            line={"color": "#ff79c6", "width": 2},
        ),
        row=2,
        col=1,
    )
    for level in (15, 75):
        valuation_figure.add_hline(
            y=level,
            line_dash="dot",
            line_color="#8793a3",
            row=2,
            col=1,
        )
    valuation_figure.update_yaxes(title_text="SPY price", row=1, col=1, secondary_y=False)
    valuation_figure.update_yaxes(
        title_text="SPY / Gold × 100", row=1, col=1, secondary_y=True
    )
    valuation_figure.update_yaxes(title_text="Normalized index", range=[0, 100], row=2, col=1)
    valuation_figure.update_xaxes(
        range=[DISPLAY_FROM, valuation_pair.index[-1]], row=1, col=1
    )
    valuation_figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    write_figure(valuation_figure, "valuation-index.html", "Williams Valuation Index · weekly")

    oi_column = next(
        (
            column
            for column in spy.columns
            if column.lower().replace(" ", "_") in {"open_interest", "openinterest", "oi"}
        ),
        None,
    )
    visible = (dates >= DISPLAY_FROM).to_numpy()
    poiv_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        subplot_titles=("SPY daily OHLC candles", "Available volume input"),
    )
    add_candles(
        poiv_figure,
        dates[visible],
        open_[visible],
        high[visible],
        low[visible],
        close[visible],
        name="SPY OHLC",
        row=1,
        col=1,
    )
    if oi_column is not None:
        open_interest = spy[oi_column].to_numpy(dtype=np.float64)
        poiv = williams_poiv_ad(high, low, close, volume, open_interest)
        poiv_figure.add_trace(
            go.Scatter(
                x=dates[visible],
                y=poiv[visible],
                mode="lines",
                name="Williams POIV AD",
                line={"color": "#ff79c6", "width": 1.8},
            ),
            row=2,
            col=1,
        )
        poiv_figure.update_yaxes(title_text="POIV AD", row=2, col=1)
    else:
        poiv_figure.add_trace(
            go.Bar(
                x=dates[visible],
                y=volume[visible],
                name="Volume",
                marker_color="#57c7ff",
                opacity=0.75,
            ),
            row=2,
            col=1,
        )
        poiv_figure.add_annotation(
            text="Open interest unavailable · POIV not calculated",
            x=0.5,
            y=0.92,
            xref="paper",
            yref="paper",
            showarrow=False,
            bgcolor="#7a2431",
            borderpad=8,
        )
        poiv_figure.update_yaxes(title_text="Volume", row=2, col=1)
    poiv_figure.update_yaxes(title_text="Price", row=1, col=1)
    poiv_figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    write_figure(poiv_figure, "poiv-ad-inputs.html", "Williams POIV AD · input availability")

    blast_off = blast_off_index(open_, high, low, close, period=1)
    write_figure(
        price_indicator_figure(
            dates, open_, high, low, close, blast_off, "Blast Off! · 1 bar", "#ff8f70"
        ),
        "blast-off.html",
        "Blast Off! · SPY daily",
    )

    trend_line, trend_direction = willtrend(high, low, close, period=66, multiplier=2.236)
    trend_visible = (dates >= DISPLAY_FROM).to_numpy()
    trend_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        subplot_titles=("SPY daily candles and WillTrend line", "Trend direction"),
    )
    add_candles(
        trend_figure,
        dates[trend_visible],
        open_[trend_visible],
        high[trend_visible],
        low[trend_visible],
        close[trend_visible],
        name="SPY OHLC",
        row=1,
        col=1,
    )
    trend_figure.add_trace(
        go.Scatter(
            x=dates[trend_visible],
            y=trend_line[trend_visible],
            mode="lines",
            name="WillTrend",
            line={"color": "#ffb86c", "width": 2},
        ),
        row=1,
        col=1,
    )
    trend_figure.add_trace(
        go.Scatter(
            x=dates[trend_visible],
            y=trend_direction[trend_visible],
            mode="lines",
            name="Direction (+1 / −1)",
            line={"color": "#62d6a5", "width": 1.6},
        ),
        row=2,
        col=1,
    )
    trend_figure.update_yaxes(title_text="Price", row=1, col=1)
    trend_figure.update_yaxes(
        title_text="Direction", range=[-1.2, 1.2], tickvals=[-1, 0, 1], row=2, col=1
    )
    trend_figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    write_figure(trend_figure, "willtrend.html", "WillTrend · SPY daily")

    swing_figure = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("SPY daily candles · confirmed ringed swings",),
    )
    add_candles(
        swing_figure,
        dates[visible],
        open_[visible],
        high[visible],
        low[visible],
        close[visible],
        name="SPY OHLC",
        row=1,
        col=1,
    )
    high_count, low_count = add_ringed_swing_arrows(
        swing_figure, dates, high, low, visible
    )
    visible_high = high[visible]
    visible_low = low[visible]
    price_padding = 0.06 * (visible_high.max() - visible_low.min())
    swing_figure.update_yaxes(
        title_text="Price",
        range=[visible_low.min() - price_padding, visible_high.max() + price_padding],
        row=1,
        col=1,
    )
    swing_figure.update_xaxes(rangeslider_visible=False, row=1, col=1)
    write_figure(
        swing_figure,
        "ringed-swing-points.html",
        "Universal Entry Technique · ringed swing points",
    )

    print(f"Wrote Plotly charts to {OUTPUT_DIR}")
    print(f"SPY bars: {len(spy):,} ({dates.iloc[0].date()}–{dates.iloc[-1].date()})")
    print(f"Gold COT reports: {len(gold_cot):,}; SPY open interest field: {oi_column or 'none'}")
    print(f"Visible SPY ringed swings: {high_count:,} highs, {low_count:,} lows")


if __name__ == "__main__":
    build_charts()
