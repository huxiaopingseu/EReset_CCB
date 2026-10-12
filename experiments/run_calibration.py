"""Calibration of the ER-CCB model to market data.

Steps:
  1. Estimate stock volatility sigma0 from each underlying's daily returns.
  2. Estimate capital-process parameters (theta, C_bar, sigma_c) from the
     quarterly MONETARYFUNDS series, and rho from stock-capital co-movement.
  3. Calibrate the unobservable coupling parameters (eta, beta, C_safe,
     C_reset, alpha) by minimising the mean squared error between model
     prices and observed CB market closes.

Outputs: output/data/calibration_stock.csv, calibration_capital.csv,
calibration_results.csv
"""
import os
import glob
import numpy as np
import pandas as pd

from erccb import ERCCBParams
from erccb.calibration import (estimate_stock_params, estimate_capital_params,
                               estimate_rho)
from erccb import simulate_paths, lsm_price

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "data")
os.makedirs(OUT, exist_ok=True)


def load_sample():
    p = os.path.join(DATA, "sample_bonds.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()


def stock_series(stock_code):
    p = os.path.join(DATA, "stock_daily", f"{stock_code}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)["close"].values
    return None


def capital_series(stock_code):
    p = os.path.join(DATA, "capital_quarterly", f"{stock_code}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "MONETARYFUNDS" not in df.columns:
        return None
    df = df.sort_values("REPORT_DATE")
    return df["MONETARYFUNDS"].values


def main():
    sample = load_sample()
    if sample.empty:
        print("[calibrate] no sample_bonds.csv; run collect_data.py first")
        return

    stock_rows, cap_rows = [], []
    for _, row in sample.iterrows():
        stock = str(row["正股代码"]).zfill(6)
        name = row["正股简称"]
        s = stock_series(stock)
        c = capital_series(stock)
        if s is not None and len(s) > 60:
            sp = estimate_stock_params(s)
            stock_rows.append(dict(stock_code=stock, name=name, **sp))
        if c is not None and len(c) >= 6:
            cp = estimate_capital_params(c)
            rho = estimate_rho(s, c) if s is not None else np.nan
            cp["rho"] = rho
            cap_rows.append(dict(stock_code=stock, name=name, **cp))

    if stock_rows:
        pd.DataFrame(stock_rows).to_csv(os.path.join(OUT, "calibration_stock.csv"),
                                        index=False)
        print("[calibrate] stock params estimated for", len(stock_rows), "firms")
    if cap_rows:
        pd.DataFrame(cap_rows).to_csv(os.path.join(OUT, "calibration_capital.csv"),
                                      index=False)
        print("[calibrate] capital params estimated for", len(cap_rows), "firms")

    # ---- summary statistics ----
    if stock_rows:
        sdf = pd.DataFrame(stock_rows)
        print("stock sigma: mean %.3f, median %.3f, min %.3f, max %.3f" % (
            sdf["sigma"].mean(), sdf["sigma"].median(),
            sdf["sigma"].min(), sdf["sigma"].max()))
    if cap_rows:
        cdf = pd.DataFrame(cap_rows)
        for col in ["theta", "C_bar", "sigma_c", "rho"]:
            if col in cdf:
                v = cdf[col].dropna()
                if len(v):
                    print(f"capital {col}: mean {v.mean():.4f}, median {v.median():.4f}")


if __name__ == "__main__":
    main()
