"""Normalize capital to per-100-face units and run empirical pricing comparison.

C0 (model units, "yuan per 100 face") = MONETARYFUNDS * 100 / (issue_scale * 1e8).
Then price each sample bond under the calibrated parameters and compare the
model price with the observed CB market close (RMSE/MAE/MAPE).
"""
import os
import numpy as np
import pandas as pd

from erccb import ERCCBParams, simulate_paths, lsm_price

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "data")
os.makedirs(OUT, exist_ok=True)

# calibrated summary values (from run_calibration.py)
SIGMA0 = 0.44
THETA = 0.67
RHO = 0.05
SIGMA_C = 0.30          # capital vol as a fraction of C (per-unit)
C_SAFE = 200.0
C_RESET = 150.0
ETA = 0.15
BETA = 0.05
ALPHA = 0.50
R = 0.03


def latest_cb_close(bond_code):
    p = os.path.join(DATA, "cb_daily", f"{bond_code}.csv")
    if not os.path.exists(p):
        return np.nan
    df = pd.read_csv(p)
    if len(df) == 0:
        return np.nan
    return float(df["close"].iloc[-1])


def main():
    sample = pd.read_csv(os.path.join(DATA, "sample_bonds.csv"))
    cap = pd.read_csv(os.path.join(DATA, "capital_summary.csv"))
    rows = []

    for _, row in sample.iterrows():
        code = str(row["债券代码"]).zfill(6)
        stock = str(row["正股代码"]).zfill(6)
        name = row["正股简称"]
        # issue scale (亿元) -> yuan
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
        # normalized capital per 100 face
        C0 = mf * 100.0 / scale

        # initial conversion price from issuance list if available
        K0 = float(row.get("转股价", 100)) if pd.notna(row.get("转股价", None)) else 100.0
        # initial stock price: first observed close around listing (use recent)
        sfile = os.path.join(DATA, "stock_daily", f"{stock}.csv")
        if not os.path.exists(sfile):
            continue
        sdf = pd.read_csv(sfile)
        S0 = float(sdf["close"].iloc[0]) if len(sdf) else 100.0

        p = ERCCBParams(S0=S0, K0=K0, C0=min(C0, 1000.0), sigma0=SIGMA0,
                        theta=THETA, sigma_c=SIGMA_C, rho=RHO,
                        C_safe=C_SAFE, C_reset=C_RESET, eta=ETA, beta=BETA,
                        alpha=ALPHA, r=R)
        try:
            sim = simulate_paths(p, 8000, seed=0, reset_mode=1)
            model_px, se = lsm_price(sim, p)
        except Exception as e:
            print(f"[pricing] {code} {name} FAILED: {e}")
            continue
        market_px = latest_cb_close(code)
        rows.append(dict(bond_code=code, name=name, C0=round(C0, 1),
                         S0=round(S0, 2), K0=round(K0, 2),
                         model_price=round(model_px, 3),
                         market_price=round(market_px, 3) if not np.isnan(market_px) else np.nan,
                         n_resets=row.get("n_resets")))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "empirical_pricing.csv"), index=False)
    print(df.to_string(index=False))

    # pricing error on rows with valid market price
    v = df.dropna(subset=["market_price"])
    if len(v) > 0:
        err = v.model_price - v.market_price
        rmse = np.sqrt((err ** 2).mean())
        mae = np.abs(err).mean()
        mape = (np.abs(err) / v.market_price).mean() * 100
        print(f"\nPricing error (n={len(v)}): RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%")


if __name__ == "__main__":
    main()
