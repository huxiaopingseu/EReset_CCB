"""Parameter calibration for the ER-CCB model.

Stock parameters are estimated from the underlying stock's daily returns
(historical volatility / MLE for a geometric Brownian motion).  Capital
parameters (mean-reversion speed theta, long-run level C_bar, volatility
sigma_c) are estimated from the quarterly operating-capital series with a
discrete-time mean-reverting model; the stock-capital correlation rho is
estimated from overlapping returns.  The unobservable coupling parameters
(C_safe, C_reset, eta, beta, alpha) are calibrated by minimising the
mean-squared pricing error against observed convertible-bond market prices.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_stock_params(close, dt=1.0 / 252.0):
    """Estimate (sigma, mu) of a GBM from daily closes via log-returns.

    Returns dict with sigma (annualised), mu (annualised), n.
    """
    r = np.diff(np.log(np.asarray(close, dtype=float)))
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return dict(sigma=np.nan, mu=np.nan, n=len(r))
    sigma_daily = r.std(ddof=1)
    mu_daily = r.mean()
    sigma = sigma_daily / np.sqrt(dt)
    mu = mu_daily / dt
    return dict(sigma=sigma, mu=mu, n=len(r))


def estimate_capital_params(C_series, dt_quarter=0.25):
    """Fit a discrete AR(1) (mean-reverting) process to the capital series.

    Model: C_{t+1} = (1 - theta*dt) C_t + theta*dt*C_bar + noise,
    so regression C_{t+1} = a + b C_t gives theta = (1-b)/dt, C_bar = a/(theta*dt),
    and sigma_c from the residual volatility scaled to annual units.
    Returns dict(theta, C_bar, sigma_c, r_squared, n).
    """
    C = np.asarray(C_series, dtype=float)
    C = C[np.isfinite(C)]
    if len(C) < 6:
        return dict(theta=np.nan, C_bar=np.nan, sigma_c=np.nan, r_squared=np.nan, n=len(C))
    x = C[:-1]
    y = C[1:]
    # OLS
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    theta = (1.0 - b) / dt_quarter
    if theta <= 0:
        theta = np.nan
        C_bar = np.nan
    else:
        C_bar = a / (theta * dt_quarter)
    resid = y - (a + b * x)
    sigma_c = resid.std(ddof=1) / np.sqrt(dt_quarter)  # annualised (per unit C)
    # residual relative volatility (sigma_c as fraction of capital, if C_bar>0)
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(theta=theta, C_bar=C_bar, sigma_c=sigma_c, r_squared=r2, n=len(C))


def estimate_rho(stock_close, C_series):
    """Estimate stock-capital correlation from overlapping quarterly returns."""
    C = np.asarray(C_series, dtype=float)
    s = np.asarray(stock_close, dtype=float)
    n = min(len(C), len(s))
    if n < 6:
        return np.nan
    dC = np.diff(C[:n])
    dS = np.diff(np.log(s[:n]))
    m = min(len(dC), len(dS))
    dC, dS = dC[:m], dS[:m]
    if m < 4 or dC.std() == 0 or dS.std() == 0:
        return np.nan
    return float(np.corrcoef(dC, dS)[0, 1])


def calibrate_coupling(params, bond_codes, stock_daily_dir, cb_daily_dir,
                       fixed=None, bounds=None, N=8000, seed=0):
    """Calibrate unobservable coupling params by minimising pricing MSE.

    For each bond in the sample, the model price is computed with the given
    coupling parameters (and stock/capital params estimated from data); the
    objective is the mean squared error between model and market CB price.
    """
    # This is the heavy, dataset-dependent routine; implemented in
    # experiments/run_calibration.py where data paths are known.
    raise NotImplementedError("see experiments/run_calibration.py")
