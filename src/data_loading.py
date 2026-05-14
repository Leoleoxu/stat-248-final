"""Data loading and feature engineering for Apple quarterly returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "apple_financial_dataset"
MASTER_FILE = DATA_DIR / "aapl_quarterly_master.csv"
SUMMARY_FILE = DATA_DIR / "aapl_quarterly_summary.csv"

QUARTER_TO_INT = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def _build_period_index(df: pd.DataFrame) -> pd.PeriodIndex:
    quarters = df["fiscal_quarter"].map(QUARTER_TO_INT)
    return pd.PeriodIndex.from_fields(
        year=df["fiscal_year"].to_numpy(),
        quarter=quarters.to_numpy(),
        freq="Q",
    )


def load_quarterly_master() -> pd.DataFrame:
    """Load the primary quarterly modeling table indexed by fiscal quarter."""
    df = pd.read_csv(MASTER_FILE)
    df["quarter_start"] = pd.to_datetime(df["quarter_start"])
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df.index = _build_period_index(df)
    df.index.name = "period"
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("Duplicate fiscal periods detected in master file.")
    return df


def load_quarterly_summary() -> pd.DataFrame:
    """Load the supplementary quarterly summary with revenue-share covariates."""
    df = pd.read_csv(SUMMARY_FILE)
    df.index = _build_period_index(df)
    df.index.name = "period"
    return df.sort_index()


def get_target(target: str = "quarter_return") -> pd.Series:
    """Return the target series indexed by fiscal quarter."""
    df = load_quarterly_master()
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in master columns.")
    return df[target].rename(target)


def make_lag_features(
    y: pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    add_quarter_dummies: bool = True,
    add_trend: bool = False,
) -> pd.DataFrame:
    """Build a design matrix of lagged values plus optional seasonal dummies.

    Rows where any required lag is NaN are dropped, so the returned frame is
    aligned with a target that itself is lagged into the future.
    """
    frame = pd.DataFrame(index=y.index)
    for lag in lags:
        frame[f"lag{lag}"] = y.shift(lag)
    if add_quarter_dummies:
        quarter = y.index.quarter
        # Drop Q1 to avoid the dummy trap; intercept absorbs the baseline.
        for q in (2, 3, 4):
            frame[f"q{q}"] = (quarter == q).astype(int)
    if add_trend:
        frame["trend"] = np.arange(len(frame))
    return frame.dropna()


def aligned_xy(
    y: pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    add_quarter_dummies: bool = True,
    add_trend: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned on the same index after dropping rows with NaN lags."""
    X = make_lag_features(
        y, lags=lags, add_quarter_dummies=add_quarter_dummies, add_trend=add_trend
    )
    y_aligned = y.loc[X.index]
    return X, y_aligned


def merge_revenue_shares(master: pd.DataFrame) -> pd.DataFrame:
    """Attach product revenue-share columns from the summary table."""
    summary = load_quarterly_summary()
    share_cols = [c for c in summary.columns if c.startswith("share_")]
    return master.join(summary[share_cols], how="left")


def chronological_split(
    series: pd.Series, test_start: str
) -> tuple[pd.Series, pd.Series]:
    """Split a quarter-indexed series into train (< test_start) and test (>= test_start).

    `test_start` is parsed as a quarterly Period string, e.g. '2023Q1'.
    """
    cutoff = pd.Period(test_start, freq="Q")
    train = series[series.index < cutoff]
    test = series[series.index >= cutoff]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Empty split at {test_start}: train={len(train)}, test={len(test)}.")
    return train, test
