"""Forecasting models: naive baselines, ARIMA/SARIMA, and lagged regressions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_loading import aligned_xy


# ---------- Naive baselines ----------

def naive_historical_mean(train: pd.Series, test_index: pd.PeriodIndex) -> pd.Series:
    """Constant forecast equal to the training mean."""
    return pd.Series(train.mean(), index=test_index, name="naive_mean")


def naive_last_value(train: pd.Series, test: pd.Series) -> pd.Series:
    """One-step persistence: forecast for t is the actual at t-1.

    Uses true past values (not previous forecasts) so it remains a one-step-ahead
    benchmark identical in fairness to the ARIMA one-step recursion.
    """
    history = pd.concat([train, test])
    fc = history.shift(1).loc[test.index]
    return fc.rename("naive_last")


def naive_seasonal(train: pd.Series, test: pd.Series, season: int = 4) -> pd.Series:
    """Seasonal naive: forecast for t is value at t - season."""
    history = pd.concat([train, test])
    fc = history.shift(season).loc[test.index]
    return fc.rename("naive_seasonal4")


# ---------- ARIMA / SARIMA ----------

@dataclass
class ARIMAFitResult:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    aic: float
    bic: float
    fitted: object  # statsmodels results object


def fit_arima(
    train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None = None,
) -> ARIMAFitResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if seasonal_order is None:
            model = ARIMA(train.values, order=order)
            fitted = model.fit()
        else:
            model = SARIMAX(
                train.values,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
    return ARIMAFitResult(order, seasonal_order, fitted.aic, fitted.bic, fitted)


def arima_grid_search(
    train: pd.Series,
    p_range: range = range(0, 4),
    d_range: range = range(0, 2),
    q_range: range = range(0, 4),
    criterion: str = "aic",
) -> pd.DataFrame:
    """Exhaustive ARIMA(p,d,q) grid; returns sorted summary table."""
    rows = []
    for p, d, q in product(p_range, d_range, q_range):
        if p == 0 and q == 0 and d == 0:
            continue
        try:
            res = fit_arima(train, (p, d, q))
            rows.append({"p": p, "d": d, "q": q, "aic": res.aic, "bic": res.bic})
        except Exception as exc:  # noqa: BLE001
            rows.append({"p": p, "d": d, "q": q, "aic": np.nan, "bic": np.nan, "error": str(exc)})
    return pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)


def sarima_grid_search(
    train: pd.Series,
    p_range: range = range(0, 3),
    d_range: range = range(0, 2),
    q_range: range = range(0, 3),
    P_range: range = range(0, 2),
    D_range: range = range(0, 2),
    Q_range: range = range(0, 2),
    s: int = 4,
    criterion: str = "aic",
) -> pd.DataFrame:
    rows = []
    for p, d, q, P, D, Q in product(p_range, d_range, q_range, P_range, D_range, Q_range):
        if (p, d, q, P, D, Q) == (0, 0, 0, 0, 0, 0):
            continue
        try:
            res = fit_arima(train, (p, d, q), seasonal_order=(P, D, Q, s))
            rows.append(
                {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s,
                 "aic": res.aic, "bic": res.bic}
            )
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)


def arima_one_step_forecast(
    fit_result: ARIMAFitResult,
    train: pd.Series,
    test: pd.Series,
) -> pd.Series:
    """One-step-ahead forecasts on the test period, refitting parameters each step.

    Conservative approach: re-estimate the model after each new observation is
    revealed, mimicking a forecaster who updates with the latest data.
    """
    history = list(train.values)
    preds = []
    for true_value in test.values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fit_result.seasonal_order is None:
                m = ARIMA(history, order=fit_result.order).fit()
            else:
                m = SARIMAX(
                    history,
                    order=fit_result.order,
                    seasonal_order=fit_result.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
        forecast = m.forecast(steps=1)
        fc_value = float(forecast.iloc[0]) if hasattr(forecast, "iloc") else float(np.atleast_1d(forecast)[0])
        preds.append(fc_value)
        history.append(true_value)
    name = f"ARIMA{fit_result.order}"
    if fit_result.seasonal_order is not None:
        name = f"SARIMA{fit_result.order}x{fit_result.seasonal_order}"
    return pd.Series(preds, index=test.index, name=name)


# ---------- Regression with lagged features ----------

@dataclass
class RegressionFit:
    name: str
    estimator: object
    scaler: StandardScaler
    alpha: float | None
    coefficients: pd.Series


def _scale_fit(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def fit_ols(X_train: pd.DataFrame, y_train: pd.Series) -> RegressionFit:
    scaler = _scale_fit(X_train)
    est = LinearRegression()
    est.fit(scaler.transform(X_train.values), y_train.values)
    coefs = pd.Series(est.coef_, index=X_train.columns)
    return RegressionFit("OLS", est, scaler, None, coefs)


def _tune_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator_cls,
    alphas: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, pd.DataFrame]:
    """Time-aware alpha tuning with TimeSeriesSplit."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    scaler = _scale_fit(X_train)
    Xs = scaler.transform(X_train.values)
    rows = []
    for alpha in alphas:
        rmses = []
        for tr_idx, va_idx in cv.split(Xs):
            est = estimator_cls(alpha=alpha, max_iter=20000)
            est.fit(Xs[tr_idx], y_train.values[tr_idx])
            pred = est.predict(Xs[va_idx])
            rmses.append(np.sqrt(np.mean((pred - y_train.values[va_idx]) ** 2)))
        rows.append({"alpha": alpha, "rmse": np.mean(rmses)})
    df = pd.DataFrame(rows)
    best = df.loc[df["rmse"].idxmin(), "alpha"]
    return float(best), df


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: np.ndarray | None = None,
    n_splits: int = 5,
) -> tuple[RegressionFit, pd.DataFrame]:
    if alphas is None:
        alphas = np.logspace(-3, 3, 25)
    best_alpha, cv_table = _tune_alpha(X_train, y_train, Ridge, alphas, n_splits)
    scaler = _scale_fit(X_train)
    est = Ridge(alpha=best_alpha)
    est.fit(scaler.transform(X_train.values), y_train.values)
    coefs = pd.Series(est.coef_, index=X_train.columns)
    return RegressionFit("Ridge", est, scaler, best_alpha, coefs), cv_table


def fit_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: np.ndarray | None = None,
    n_splits: int = 5,
) -> tuple[RegressionFit, pd.DataFrame]:
    if alphas is None:
        alphas = np.logspace(-4, 1, 25)
    best_alpha, cv_table = _tune_alpha(X_train, y_train, Lasso, alphas, n_splits)
    scaler = _scale_fit(X_train)
    est = Lasso(alpha=best_alpha, max_iter=20000)
    est.fit(scaler.transform(X_train.values), y_train.values)
    coefs = pd.Series(est.coef_, index=X_train.columns)
    return RegressionFit("LASSO", est, scaler, best_alpha, coefs), cv_table


def regression_predict(fit: RegressionFit, X: pd.DataFrame) -> pd.Series:
    Xs = fit.scaler.transform(X.values)
    return pd.Series(fit.estimator.predict(Xs), index=X.index, name=fit.name)


def build_lagged_design(
    y: pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    add_quarter_dummies: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper around data_loading.aligned_xy."""
    return aligned_xy(y, lags=lags, add_quarter_dummies=add_quarter_dummies)
