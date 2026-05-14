"""Forecast evaluation metrics, rolling-origin backtest, and Diebold-Mariano test."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


# ---------- Metrics ----------

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true.values - y_pred.values)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true.values - y_pred.values) ** 2)))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric MAPE in percent. Returns NaN if denominator is zero everywhere."""
    denom = (np.abs(y_true.values) + np.abs(y_pred.values)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    vals = np.abs(y_true.values[mask] - y_pred.values[mask]) / denom[mask]
    return float(100.0 * np.mean(vals))


def mape_filtered(y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.05) -> float:
    """Standard MAPE in percent computed only on observations with |y_true| >= threshold."""
    mask = np.abs(y_true.values) >= threshold
    if mask.sum() == 0:
        return float("nan")
    vals = np.abs((y_true.values[mask] - y_pred.values[mask]) / y_true.values[mask])
    return float(100.0 * np.mean(vals))


def metric_table(y_true: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute MAE, RMSE, sMAPE, and filtered-MAPE for each forecast."""
    rows = {}
    for name, fc in forecasts.items():
        yt = y_true.loc[fc.index].dropna()
        fc_clean = fc.loc[yt.index]
        rows[name] = {
            "MAE": mae(yt, fc_clean),
            "RMSE": rmse(yt, fc_clean),
            "sMAPE": smape(yt, fc_clean),
            "MAPE_filtered": mape_filtered(yt, fc_clean),
            "n": len(yt),
        }
    return pd.DataFrame(rows).T


def add_relative_improvement(
    table: pd.DataFrame, baseline: str, metric: str = "RMSE"
) -> pd.DataFrame:
    """Append percent improvement of each row over a baseline row."""
    if baseline not in table.index:
        raise KeyError(f"Baseline '{baseline}' not in table.")
    base = table.loc[baseline, metric]
    out = table.copy()
    out[f"{metric}_vs_{baseline}_%"] = 100.0 * (base - out[metric]) / base
    return out


# ---------- Diebold-Mariano test ----------

def diebold_mariano(
    y_true: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    h: int = 1,
    loss: str = "squared",
) -> dict:
    """Two-sided DM test with the Harvey-Leybourne-Newbold small-sample correction.

    H0: equal predictive accuracy (E[d_t] = 0).
    Returns the DM statistic, HLN-corrected statistic, and two-sided p-values.
    """
    idx = y_true.index.intersection(pred_a.index).intersection(pred_b.index)
    e_a = (y_true.loc[idx] - pred_a.loc[idx]).values
    e_b = (y_true.loc[idx] - pred_b.loc[idx]).values
    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"Unknown loss '{loss}'.")

    n = len(d)
    d_bar = float(np.mean(d))
    # Long-run variance via sum of autocovariances up to h-1
    gamma0 = float(np.var(d, ddof=0))
    lr_var = gamma0
    for k in range(1, h):
        gk = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
        lr_var += 2.0 * gk
    if lr_var <= 0 or n == 0:
        return {"dm": np.nan, "p_value": np.nan, "hln": np.nan, "p_hln": np.nan, "n": n}
    dm_stat = d_bar / np.sqrt(lr_var / n)
    p_dm = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    # HLN correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln_stat = dm_stat * correction
    p_hln = 2.0 * (1.0 - stats.t.cdf(abs(hln_stat), df=n - 1))
    return {"dm": float(dm_stat), "p_value": float(p_dm),
            "hln": float(hln_stat), "p_hln": float(p_hln), "n": n}


def dm_against_baseline(
    y_true: pd.Series,
    forecasts: dict[str, pd.Series],
    baseline: str,
    h: int = 1,
) -> pd.DataFrame:
    rows = []
    base = forecasts[baseline]
    for name, fc in forecasts.items():
        if name == baseline:
            continue
        res = diebold_mariano(y_true, fc, base, h=h)
        rows.append({"model": name, **res})
    return pd.DataFrame(rows).set_index("model")


# ---------- Rolling-origin backtest ----------

@dataclass
class RollingResult:
    name: str
    forecasts: pd.Series  # one-step predictions over the backtest window
    actuals: pd.Series


def rolling_one_step(
    y: pd.Series,
    forecast_fn: Callable[[pd.Series], float],
    start_period: pd.Period,
    name: str,
) -> RollingResult:
    """Generic rolling-origin one-step forecast.

    `forecast_fn(history)` is given all data strictly before the target quarter
    and must return a scalar forecast for the next quarter.
    """
    if start_period not in y.index:
        raise ValueError(f"Start period {start_period} not in index.")
    idx = y.index
    start_pos = idx.get_loc(start_period)
    out_idx, out_vals = [], []
    for pos in range(start_pos, len(idx)):
        history = y.iloc[:pos]
        try:
            fc = float(forecast_fn(history))
        except Exception:
            fc = np.nan
        out_idx.append(idx[pos])
        out_vals.append(fc)
    forecasts = pd.Series(out_vals, index=pd.PeriodIndex(out_idx, freq="Q"), name=name)
    actuals = y.loc[forecasts.index]
    return RollingResult(name, forecasts, actuals)


def rolling_metrics(results: dict[str, RollingResult]) -> pd.DataFrame:
    actual = next(iter(results.values())).actuals
    return metric_table(actual, {k: r.forecasts for k, r in results.items()})


def make_arima_forecaster(order: tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    def _fn(history: pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ARIMA(history.values, order=order).fit()
        forecast = m.forecast(steps=1)
        return float(forecast.iloc[0]) if hasattr(forecast, "iloc") else float(np.atleast_1d(forecast)[0])

    return _fn


def make_sarima_forecaster(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def _fn(history: pd.Series) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(
                history.values,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        forecast = m.forecast(steps=1)
        return float(forecast.iloc[0]) if hasattr(forecast, "iloc") else float(np.atleast_1d(forecast)[0])

    return _fn


def make_naive_mean_forecaster():
    def _fn(history: pd.Series) -> float:
        return float(history.mean())
    return _fn


def make_naive_last_forecaster():
    def _fn(history: pd.Series) -> float:
        return float(history.iloc[-1])
    return _fn


def make_naive_seasonal_forecaster(season: int = 4):
    def _fn(history: pd.Series) -> float:
        if len(history) < season:
            return float(history.mean())
        return float(history.iloc[-season])
    return _fn


def make_regression_forecaster(
    fit_fn: Callable,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    add_quarter_dummies: bool = True,
    refit_every: int = 1,
):
    """Build a one-step forecaster that refits a regression model on rolling history.

    `fit_fn(X_train, y_train)` should return a `RegressionFit` (or a tuple whose
    first element is one — Ridge/LASSO return `(fit, cv_table)`).
    """
    from data_loading import aligned_xy
    from models import regression_predict

    cache = {"step": 0, "fit": None}

    def _fn(history: pd.Series) -> float:
        X, y_aligned = aligned_xy(history, lags=lags, add_quarter_dummies=add_quarter_dummies)
        if len(X) == 0:
            return float(history.mean())
        if cache["fit"] is None or cache["step"] % refit_every == 0:
            result = fit_fn(X, y_aligned)
            cache["fit"] = result[0] if isinstance(result, tuple) else result
        cache["step"] += 1
        # Build the row of features used to predict the *next* quarter
        next_row = {}
        for lag in lags:
            next_row[f"lag{lag}"] = history.iloc[-lag]
        if add_quarter_dummies:
            next_quarter = (history.index[-1] + 1).quarter
            for q in (2, 3, 4):
                next_row[f"q{q}"] = int(next_quarter == q)
        x_next = pd.DataFrame([next_row], columns=X.columns,
                              index=[history.index[-1] + 1])
        pred = regression_predict(cache["fit"], x_next)
        return float(pred.iloc[0])

    return _fn
