"""Empirical summary: calibration statistics and reset-boundary validation.

The distinctive empirical prediction of the model is that the reset fires only
when the firm is liquidity-constrained.  We test this by computing, for each
reset firm, the normalized operating-capital coverage

    C0 = MONETARYFUNDS * 100 / issue_scale     (yuan per 100 face),

and comparing it against the model's strategic threshold C_reset.  Firms with
C0 below the threshold should reset immediately on a price trigger; firms with
ample capital can "hold out" even when the 15/30 condition is met.

Outputs: output/data/empirical_capital_coverage.csv and a summary printout.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "data")
os.makedirs(OUT, exist_ok=True)


def main():
    sample = pd.read_csv(os.path.join(DATA, "sample_bonds.csv"))
    cap = pd.read_csv(os.path.join(DATA, "capital_summary.csv"))

    rows = []
    for _, row in sample.iterrows():
        code = str(row["债券代码"]).zfill(6)
        name = row["正股简称"]
        try:
            scale = float(row["发行规模"]) * 1e8
        except (TypeError, ValueError):
            continue
        c = cap[cap.bond_code.astype(str) == code]
        if c.empty:
            continue
        mf = c.iloc[0].get("monetary_funds")
        if pd.isna(mf) or mf <= 0:
            continue
        C0 = mf * 100.0 / scale
        rows.append(dict(bond_code=code, name=name,
                         C0=round(C0, 1),
                         n_resets=int(row.get("n_resets", 1))))

    df = pd.DataFrame(rows).sort_values("C0")
    df.to_csv(os.path.join(OUT, "empirical_capital_coverage.csv"), index=False)
    print(df.to_string(index=False))

    C_reset = 150.0
    below = df[df.C0 < C_reset]
    print(f"\nSample size: {len(df)} reset firms")
    print(f"Capital coverage C0: min={df.C0.min():.0f}, median={df.C0.median():.0f}, "
          f"max={df.C0.max():.0f}")
    print(f"Firms with C0 < C_reset={C_reset:.0f}: {len(below)} "
          f"({100*len(below)/len(df):.0f}%)")
    # multi-reset firms tend to be more constrained
    multi = df[df.n_resets >= 2]
    if len(multi):
        print(f"Multi-reset firms ({len(multi)}): median C0 = {multi.C0.median():.0f}")
    single = df[df.n_resets == 1]
    if len(single):
        print(f"Single-reset firms ({len(single)}): median C0 = {single.C0.median():.0f}")


if __name__ == "__main__":
    main()
