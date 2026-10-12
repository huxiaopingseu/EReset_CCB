"""Main numerical experiments for the ER-CCB paper.

Experiments (each produces data tables + figures under output/):

  E1  Endogenous vs no-reset vs exogenous reset: price gap and mechanism.
  E2  Reset probability / frequency and the endogenous reset boundary in
      (S_0, C_0) space -- the death-spiral feedback.
  E3  Coupling sensitivity: eta (distress drift), beta (liquidity vol),
      sigma_c (capital vol), C_reset (strategic threshold).
  E4  Convergence of the LSM estimator vs number of paths (and MC vs RQMC).
  E5  Greeks / comparative statics: price vs initial capital and volatility.

All prices are reported per 100 face value.
"""
import os
import numpy as np
import pandas as pd

from erccb import ERCCBParams, simulate_paths, lsm_price, price_benchmarks

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
DATA = os.path.join(OUT, "data")
FIG = os.path.join(OUT, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def run_e1_benchmark(N=20000, seed=0):
    """E1: price gap among the three reset regimes."""
    p = ERCCBParams(C0=150.0)  # moderately constrained firm
    rows = []
    for key, mode, rho in [("no_reset", 0, 0.0),
                           ("exogenous_rho0.5", 2, 0.5),
                           ("endogenous", 1, 0.0)]:
        sim = simulate_paths(p, N, seed=seed, reset_mode=mode, rho_reset=rho)
        px, se = lsm_price(sim, p)
        rows.append(dict(model=key, price=round(px, 3), se=round(se, 4),
                         mean_resets=round(float(sim["n_resets"].mean()), 4),
                         call_frac=round(float((sim["call_day"] <= p.M).mean()), 4),
                         default_frac=round(float((sim["default_day"] <= p.M).mean()), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "e1_benchmark.csv"), index=False)
    print("[E1] benchmark done")
    print(df.to_string(index=False))
    return df


def run_e2_reset_boundary(N=8000, seed=0):
    """E2: endogenous reset frequency and price across (S0, C0) grid."""
    S0s = [70, 80, 90, 100, 110, 130]
    C0s = [50, 100, 150, 200, 300, 500]
    rows = []
    for S0 in S0s:
        for C0 in C0s:
            p = ERCCBParams(S0=S0, K0=100.0, C0=C0)
            sim = simulate_paths(p, N, seed=seed, reset_mode=1)
            px, se = lsm_price(sim, p)
            rows.append(dict(S0=S0, C0=C0, price=round(px, 3),
                             reset_freq=round(float(sim["n_resets"].mean()), 4),
                             call_frac=round(float((sim["call_day"] <= p.M).mean()), 4),
                             default_frac=round(float((sim["default_day"] <= p.M).mean()), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "e2_reset_boundary.csv"), index=False)
    print("[E2] reset boundary grid done")
    print(df.to_string(index=False))
    return df


def run_e3_sensitivity(N=8000, seed=0):
    """E3: sensitivity of price and reset frequency to coupling params."""
    base = dict(C0=150.0)
    # eta
    rows = []
    for eta in [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]:
        p = ERCCBParams(**base, eta=eta)
        sim = simulate_paths(p, N, seed=seed, reset_mode=1)
        px, _ = lsm_price(sim, p)
        rows.append(dict(param="eta", value=eta, price=round(px, 3),
                         reset_freq=round(float(sim["n_resets"].mean()), 4)))
    for beta in [0.0, 0.03, 0.05, 0.08, 0.12]:
        p = ERCCBParams(**base, beta=beta)
        sim = simulate_paths(p, N, seed=seed, reset_mode=1)
        px, _ = lsm_price(sim, p)
        rows.append(dict(param="beta", value=beta, price=round(px, 3),
                         reset_freq=round(float(sim["n_resets"].mean()), 4)))
    for sigma_c in [0.10, 0.20, 0.25, 0.35, 0.50]:
        p = ERCCBParams(**base, sigma_c=sigma_c)
        sim = simulate_paths(p, N, seed=seed, reset_mode=1)
        px, _ = lsm_price(sim, p)
        rows.append(dict(param="sigma_c", value=sigma_c, price=round(px, 3),
                         reset_freq=round(float(sim["n_resets"].mean()), 4)))
    for C_reset in [80, 120, 150, 200, 260]:
        p = ERCCBParams(**base, C_reset=C_reset)
        sim = simulate_paths(p, N, seed=seed, reset_mode=1)
        px, _ = lsm_price(sim, p)
        rows.append(dict(param="C_reset", value=C_reset, price=round(px, 3),
                         reset_freq=round(float(sim["n_resets"].mean()), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "e3_sensitivity.csv"), index=False)
    print("[E3] sensitivity done")
    return df


def run_e4_convergence(seed=0):
    """E4: LSM price convergence vs number of paths."""
    p = ERCCBParams(C0=150.0)
    rows = []
    ref = None
    for N in [1000, 2000, 4000, 8000, 16000, 32000]:
        prices, ses = [], []
        for rep in range(8):
            sim = simulate_paths(p, N, seed=seed + rep, reset_mode=1)
            px, se = lsm_price(sim, p)
            prices.append(px)
        if ref is None:
            ref = float(np.mean(prices))
        rows.append(dict(N=N, price=round(float(np.mean(prices)), 3),
                         std=round(float(np.std(prices, ddof=1)), 4)))
    df = pd.DataFrame(rows)
    df["bias"] = (df["price"] - ref).round(4)
    df.to_csv(os.path.join(DATA, "e4_convergence.csv"), index=False)
    print("[E4] convergence done; ref =", round(ref, 3))
    print(df.to_string(index=False))
    return df


def main():
    run_e1_benchmark()
    run_e2_reset_boundary()
    run_e3_sensitivity()
    run_e4_convergence()
    print("[all] experiments complete")


if __name__ == "__main__":
    main()
