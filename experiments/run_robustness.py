"""Robustness experiments for the ER-CCB paper (Section 6.5).

R1. Regression-basis robustness: price under linear / quadratic / full /
    cubic basis functions.
R2. Seed robustness: price across independent seeds (Monte Carlo noise).
R3. Fine grid on the strategic threshold C_reset (refining Table 3).

Outputs: output/data/e5_robustness_basis.csv, e5_robustness_seeds.csv,
         e5_robustness_threshold.csv
"""
import os
import numpy as np
import pandas as pd

from erccb import ERCCBParams, simulate_paths, lsm_price

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output", "data")
os.makedirs(OUT, exist_ok=True)


def run_basis(N=12000, seed=0):
    p = ERCCBParams(C0=150.0)
    sim = simulate_paths(p, N, seed=seed, reset_mode=1)
    rows = []
    for basis in ["linear", "quad", "full", "cubic"]:
        px, se = lsm_price(sim, p, basis=basis)
        rows.append(dict(basis=basis, n_terms={"linear": 3, "quad": 5,
                                               "full": 6, "cubic": 10}[basis],
                         price=round(px, 3), se=round(se, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e5_robustness_basis.csv"), index=False)
    print("[R1] basis robustness")
    print(df.to_string(index=False))
    return df


def run_seeds(N=12000, n_seeds=10):
    p = ERCCBParams(C0=150.0)
    prices = []
    for s in range(n_seeds):
        sim = simulate_paths(p, N, seed=s, reset_mode=1)
        px, _ = lsm_price(sim, p)
        prices.append(px)
    prices = np.array(prices)
    df = pd.DataFrame(dict(
        n_seeds=[n_seeds],
        mean=[round(prices.mean(), 3)],
        std=[round(prices.std(ddof=1), 4)],
        min=[round(prices.min(), 3)],
        max=[round(prices.max(), 3)],
    ))
    df.to_csv(os.path.join(OUT, "e5_robustness_seeds.csv"), index=False)
    print("[R2] seed robustness: mean %.3f  std %.4f  range [%.3f, %.3f]" % (
        prices.mean(), prices.std(ddof=1), prices.min(), prices.max()))
    return df


def run_threshold(N=10000, seed=0):
    p = ERCCBParams(C0=150.0)
    rows = []
    for C_reset in [60, 80, 100, 120, 150, 180, 220, 260, 300, 350]:
        pp = ERCCBParams(C0=150.0, C_reset=C_reset)
        sim = simulate_paths(pp, N, seed=seed, reset_mode=1)
        px, se = lsm_price(sim, pp)
        rows.append(dict(C_reset=C_reset, price=round(px, 3),
                         reset_freq=round(float(sim["n_resets"].mean()), 4),
                         se=round(se, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "e5_robustness_threshold.csv"), index=False)
    print("[R3] threshold fine grid")
    print(df.to_string(index=False))
    return df


def main():
    run_basis()
    run_seeds()
    run_threshold()
    print("[all] robustness experiments complete")


if __name__ == "__main__":
    main()
