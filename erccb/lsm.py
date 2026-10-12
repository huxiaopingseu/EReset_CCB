"""Least-squares Monte Carlo (LSM) backward induction for the ER-CCB.

The holder's American conversion right and soft-put option are priced by
regressing continuation values on a polynomial basis in (S_t, C_t, K_t).
The endogenous reset, forced call, and default are already baked into the
forward paths (simulate.py).  Coupons paid while holding are accumulated,
and early termination pays pro-rata accrued interest.
"""
import numpy as np


def _basis(S, C, K, kind="full"):
    """Polynomial basis in moneyness m = S/K and capital ratio x = C/K.

    kind: 'linear' (1,m,x), 'quad' (no cross term), 'full' (with m*x),
          'cubic' (adds cubic and mixed terms).
    """
    m = S / np.maximum(K, 1e-12)
    x = C / np.maximum(K, 1e-12)
    if kind == "linear":
        return np.column_stack([np.ones_like(m), m, x])
    if kind == "quad":
        return np.column_stack([np.ones_like(m), m, m * m, x, x * x])
    if kind == "cubic":
        return np.column_stack([
            np.ones_like(m), m, m * m, m ** 3,
            x, x * x, x ** 3,
            m * x, m * m * x, m * x * x,
        ])
    # full (default)
    return np.column_stack([
        np.ones_like(m),
        m, m * m,
        x, x * x,
        m * x,
    ])


def lsm_price(sim, params, basis="full"):
    """Price the convertible bond by LSM backward induction.

    Returns (price, standard_error).
    """
    p = params
    M = p.M
    n_days = p.n_days
    disc = np.exp(-p.r * p.dt)
    N = sim["S"].shape[0]
    F = p.F
    coupons = np.asarray(p.coupons, dtype=np.float64)

    S = sim["S"]
    C = sim["C"]
    K = sim["K"]
    put_active = sim["put_active"]
    alive = sim["alive"]
    call_day = sim["call_day"]
    default_day = sim["default_day"]

    # day index -> coupon year (1..5); final coupon c6 folded into redemption
    coupon_year = {int(round(k * n_days)): k for k in range(1, 6)}

    call_from_day = int(round(0.5 * n_days))
    put_from_day = int(round(5.0 * n_days))

    # ---- terminal cash flow (time-M money) ----
    conv_T = F * S[:, M] / np.maximum(K[:, M], 1e-12)
    redeem_T = F * (1.0 + coupons[5])
    cf = np.maximum(conv_T, redeem_T)

    # active = still holding (not yet exercised/terminated by holder choice)
    active = alive[:, M].astype(bool)

    for t in range(M - 1, -1, -1):
        # discount one step: cf is now in time-t money
        cf *= disc

        # coupon received at day t while holding
        yr = coupon_year.get(t)
        if yr is not None:
            cf[active] += coupons[yr - 1] * F

        still = alive[:, t].astype(bool) & active
        if not still.any():
            continue

        # ---- regression of continuation value on state ----
        X = _basis(S[still, t], C[still, t], K[still, t], kind=basis)
        y = cf[still]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        cont = X @ beta

        # immediate exercise values (all in time-t money)
        conv = F * S[still, t] / np.maximum(K[still, t], 1e-12)

        # pro-rata accrued interest for the current year
        year = min(int((t - 1) // n_days) + 1, 6)
        days_in = (t - 1) % n_days / float(n_days)
        accrued = coupons[year - 1] * F * days_in

        put_val = F + accrued
        exercise = np.full(still.sum(), -np.inf)

        put_ok = put_active[still, t].astype(bool) & (t >= put_from_day)
        exercise[put_ok] = np.maximum(exercise[put_ok], put_val)

        conv_ok = t >= call_from_day
        exercise[conv_ok] = np.maximum(exercise[conv_ok], conv[conv_ok])

        ex = exercise > cont
        if ex.any():
            idx = np.where(still)[0][ex]
            cf[idx] = exercise[ex]
            active[idx] = False

    # ---- adjust for paths terminated in forward sim ----
    # default: recovery of face
    defd = default_day <= M
    cf[defd] = p.recovery * F

    # forced call: conversion value + pro-rata coupon at call day (time-t money,
    # then discount back to time 0)
    called = call_day <= M
    for n in np.where(called)[0]:
        t = call_day[n]
        year = min(int((t - 1) // n_days) + 1, 6)
        days_in = (t - 1) % n_days / float(n_days)
        accrued = coupons[year - 1] * F * days_in
        cf[n] = (F * S[n, t] / max(K[n, t], 1e-12) + accrued) * (disc ** t)

    price = cf.mean()
    se = cf.std(ddof=1) / np.sqrt(N) if N > 1 else 0.0
    return float(price), float(se)
