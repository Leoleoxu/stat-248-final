"""Stationarity, autocorrelation, and residual diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    name: str
    statistic: float
    p_value: float
    n_lags: int
    critical_values: dict
    conclusion: str


def adf_test(series: pd.Series, regression: str = "c") -> StationarityResult:
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary)."""
    stat, p, n_lags, _, crit, _ = adfuller(series.dropna(), regression=regression, autolag="AIC")
    conclusion = "stationary" if p < 0.05 else "non-stationary"
    return StationarityResult("ADF", stat, p, n_lags, crit, conclusion)


def kpss_test(series: pd.Series, regression: str = "c") -> StationarityResult:
    """KPSS test. H0: stationary (around a level/trend)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, n_lags, crit = kpss(series.dropna(), regression=regression, nlags="auto")
    conclusion = "stationary" if p > 0.05 else "non-stationary"
    return StationarityResult("KPSS", stat, p, n_lags, crit, conclusion)


def stationarity_summary(series: pd.Series) -> pd.DataFrame:
    """Run ADF and KPSS and return a tidy summary table."""
    rows = []
    for fn in (adf_test, kpss_test):
        r = fn(series)
        rows.append(
            {
                "test": r.name,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "lags": r.n_lags,
                "conclusion": r.conclusion,
            }
        )
    return pd.DataFrame(rows)


def ljung_box(residuals: pd.Series, lags: tuple[int, ...] = (4, 8, 12, 16)) -> pd.DataFrame:
    """Ljung-Box test for residual autocorrelation. H0: no autocorrelation."""
    out = acorr_ljungbox(residuals.dropna(), lags=list(lags), return_df=True)
    out.index.name = "lag"
    out["white_noise"] = out["lb_pvalue"] > 0.05
    return out


def arch_test(residuals: pd.Series, lags: int = 12) -> dict:
    """Engle's ARCH test. H0: no conditional heteroskedasticity."""
    stat, p, f_stat, f_p = het_arch(residuals.dropna(), nlags=lags)
    return {
        "lm_stat": stat,
        "lm_pvalue": p,
        "f_stat": f_stat,
        "f_pvalue": f_p,
        "homoskedastic": p > 0.05,
    }


def acf_values(series: pd.Series, n_lags: int = 24) -> pd.Series:
    """Sample autocorrelations up to n_lags (excluding lag 0)."""
    from statsmodels.tsa.stattools import acf

    vals = acf(series.dropna(), nlags=n_lags, fft=True)
    return pd.Series(vals[1:], index=range(1, n_lags + 1), name="acf")


def pacf_values(series: pd.Series, n_lags: int = 24) -> pd.Series:
    """Sample partial autocorrelations up to n_lags (excluding lag 0)."""
    from statsmodels.tsa.stattools import pacf

    vals = pacf(series.dropna(), nlags=n_lags, method="ywm")
    return pd.Series(vals[1:], index=range(1, n_lags + 1), name="pacf")


def seasonal_summary(series: pd.Series) -> pd.DataFrame:
    """Mean, std, and count grouped by fiscal quarter."""
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError("series must have a PeriodIndex with quarterly frequency.")
    by_q = series.groupby(series.index.quarter)
    out = pd.DataFrame(
        {
            "mean": by_q.mean(),
            "std": by_q.std(),
            "count": by_q.count(),
        }
    )
    out.index = [f"Q{q}" for q in out.index]
    out.index.name = "fiscal_quarter"
    return out
