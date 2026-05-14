"""Plot helpers with consistent styling for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LINES = {
    "1987 crash": "1987Q4",
    "Dot-com peak": "2000Q1",
    "GFC": "2008Q4",
    "iPhone launch": "2007Q3",
    "COVID shock": "2020Q1",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 4.2),
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def _period_to_timestamp(idx: pd.PeriodIndex) -> pd.DatetimeIndex:
    return idx.to_timestamp(how="end")


def save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path)
    return path


def plot_series(
    series: pd.Series,
    title: str,
    ylabel: str = "Quarterly return",
    annotate_events: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    x = _period_to_timestamp(series.index)
    ax.plot(x, series.values, lw=1.0, color="#1f77b4")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    if annotate_events:
        for label, period_str in EVENT_LINES.items():
            try:
                p = pd.Period(period_str, freq="Q").to_timestamp(how="end")
            except Exception:
                continue
            if p < x.min() or p > x.max():
                continue
            ax.axvline(p, color="firebrick", lw=0.5, alpha=0.45)
            ax.text(p, ax.get_ylim()[1] * 0.95, label, rotation=90,
                    fontsize=7, color="firebrick", va="top", ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    return ax


def plot_seasonal_box(series: pd.Series, title: str = "Quarterly returns by fiscal quarter") -> plt.Axes:
    fig, ax = plt.subplots()
    quarter = series.index.quarter
    data = [series.values[quarter == q] for q in (1, 2, 3, 4)]
    ax.boxplot(data, tick_labels=["Q1", "Q2", "Q3", "Q4"], showmeans=True)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title(title)
    ax.set_ylabel("Return")
    return ax


def plot_rolling_stats(series: pd.Series, window: int = 12) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5.5))
    x = _period_to_timestamp(series.index)
    axes[0].plot(x, series.rolling(window).mean(), color="#1f77b4")
    axes[0].set_title(f"Rolling mean (window = {window} quarters)")
    axes[0].axhline(0, color="grey", lw=0.6, ls="--")
    axes[1].plot(x, series.rolling(window).std(), color="#d62728")
    axes[1].set_title(f"Rolling std (window = {window} quarters)")
    fig.tight_layout()
    return fig


def plot_acf_pacf(series: pd.Series, n_lags: int = 24, title_prefix: str = "") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(series.dropna(), lags=n_lags, ax=axes[0], zero=False)
    plot_pacf(series.dropna(), lags=n_lags, ax=axes[1], zero=False, method="ywm")
    axes[0].set_title(f"{title_prefix}ACF".strip())
    axes[1].set_title(f"{title_prefix}PACF".strip())
    fig.tight_layout()
    return fig


def plot_forecasts(
    truth: pd.Series,
    forecasts: dict[str, pd.Series],
    title: str = "Forecasts vs. actual",
    show_train_tail: int = 16,
    train: pd.Series | None = None,
) -> plt.Axes:
    fig, ax = plt.subplots()
    if train is not None and show_train_tail > 0:
        tail = train.iloc[-show_train_tail:]
        ax.plot(_period_to_timestamp(tail.index), tail.values,
                color="black", lw=1.0, label="train (recent)")
    ax.plot(_period_to_timestamp(truth.index), truth.values,
            color="black", lw=1.4, marker="o", ms=4, label="actual")
    palette = plt.cm.tab10.colors
    for i, (name, fc) in enumerate(forecasts.items()):
        ax.plot(_period_to_timestamp(fc.index), fc.values,
                color=palette[i % len(palette)], lw=1.1, marker="s", ms=3, label=name)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_title(title)
    ax.set_ylabel("Quarterly return")
    ax.legend(ncol=2, loc="best")
    return ax


def plot_residual_panel(residuals: pd.Series, title: str = "Residual diagnostics") -> plt.Figure:
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = _period_to_timestamp(residuals.index)
    axes[0, 0].plot(x, residuals.values, lw=1.0)
    axes[0, 0].axhline(0, color="grey", lw=0.6, ls="--")
    axes[0, 0].set_title("Residuals")

    plot_acf(residuals.dropna(), lags=min(24, len(residuals) // 4),
             ax=axes[0, 1], zero=False)
    axes[0, 1].set_title("Residual ACF")

    axes[1, 0].hist(residuals.dropna(), bins=24, color="#1f77b4", alpha=0.85)
    axes[1, 0].set_title("Residual histogram")

    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot")
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


def plot_metric_bars(
    table: pd.DataFrame,
    metric: str,
    title: str | None = None,
    highlight: str | None = None,
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_tbl = table.sort_values(metric)
    colors = ["#1f77b4"] * len(sorted_tbl)
    if highlight is not None and highlight in sorted_tbl.index:
        colors[list(sorted_tbl.index).index(highlight)] = "#d62728"
    ax.barh(sorted_tbl.index, sorted_tbl[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Model comparison ({metric})")
    return ax
