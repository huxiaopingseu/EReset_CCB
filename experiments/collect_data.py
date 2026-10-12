"""Collect market data for ER-CCB calibration and empirical pricing.

Data collected (all via akshare, public sources Eastmoney / jisilu):
  1. A sample of Chinese convertible bonds that experienced downward resets,
     joined to their underlying stock codes and issue terms.
  2. Daily closing prices of the underlying stocks.
  3. Daily closing prices of the convertible bonds (pricing-error benchmark).
  4. Quarterly balance-sheet monetary funds MONETARYFUNDS (operating-capital
     proxy C_t) and short-term borrowing SHORT_LOAN.

Outputs under data/: sample_bonds.csv, stock_daily/*.csv, cb_daily/*.csv,
capital_quarterly/*.csv, capital_summary.csv.
"""
import os
import time
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
os.makedirs(os.path.join(DATA, "stock_daily"), exist_ok=True)
os.makedirs(os.path.join(DATA, "cb_daily"), exist_ok=True)
os.makedirs(os.path.join(DATA, "capital_quarterly"), exist_ok=True)

RESET_EVENTS = "/home/xiaoping/CCB/output/data/cn_cb_reset_events.csv"
ISSUANCE = "/home/xiaoping/CCB/output/data/cn_cb_issuance_list.csv"

N_SAMPLE = 25


def retry(fn, *args, tries=4, base_delay=2.0, **kw):
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    return None


def build_sample():
    ev = pd.read_csv(RESET_EVENTS)
    ev["year"] = ev["股东大会日"].str[:4].astype(int)
    iss = pd.read_csv(ISSUANCE)
    iss_map = iss[["债券代码", "正股代码", "正股简称", "发行规模", "转股价", "信用评级"]].drop_duplicates("债券代码")
    m = ev.merge(iss_map, on="债券代码", how="left")
    m = m.dropna(subset=["正股代码"]).drop_duplicates("债券代码")
    m = m.sort_values(["year", "债券代码"])
    m["正股代码"] = m["正股代码"].astype(int).astype(str).str.zfill(6)

    agg = ev.groupby("债券代码").agg(
        n_resets=("股东大会日", "count"),
        first_reset=("股东大会日", "min"),
        last_reset=("股东大会日", "max"),
    ).reset_index()

    sample = m.merge(agg, on="债券代码", how="left")
    sample = sample[(sample["n_resets"] >= 1) & (sample["year"] >= 2018)]
    sample = sample.drop_duplicates("正股代码")
    sample = sample.sort_values("year").head(N_SAMPLE)
    return sample


def stock_code_format(code):
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return "SH" + code
    if code.startswith(("0", "3")):
        return "SZ" + code
    if code.startswith(("4", "8")):
        return "BJ" + code
    return code


def main():
    sample = build_sample()
    sample.to_csv(os.path.join(DATA, "sample_bonds.csv"), index=False)
    print(f"[sample] {len(sample)} bonds selected")

    records = []
    for i, row in sample.iterrows():
        code = str(row["债券代码"]).zfill(6)
        stock = str(row["正股代码"]).zfill(6)
        name = row["正股简称"]
        try:
            issue_scale = float(row.get("发行规模"))
        except (TypeError, ValueError):
            issue_scale = None

        # ---- 1. stock daily ----
        try:
            sdf = retry(ak.stock_zh_a_hist, symbol=stock, period="daily",
                        start_date="20190101", end_date="20261231", adjust="")
            sdf = sdf[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"})
            sdf.to_csv(os.path.join(DATA, "stock_daily", f"{stock}.csv"), index=False)
        except Exception as e:
            print(f"[stock] {stock} {name} FAILED: {e}")

        # ---- 2. CB daily ----
        try:
            sym = ("sh" if stock.startswith(("6", "9")) else "sz") + code
            cdf = retry(ak.bond_zh_hs_cov_daily, symbol=sym)
            cdf = cdf[["date", "close"]]
            cdf.to_csv(os.path.join(DATA, "cb_daily", f"{code}.csv"), index=False)
        except Exception as e:
            print(f"[cb] {code} {name} FAILED: {e}")

        # ---- 3. quarterly balance sheet ----
        try:
            sym_full = stock_code_format(stock)
            bdf = retry(ak.stock_balance_sheet_by_report_em, symbol=sym_full)
            keep = [c for c in ["REPORT_DATE", "MONETARYFUNDS", "SHORT_LOAN",
                                "TOTAL_LIABILITIES"] if c in bdf.columns]
            bdf = bdf[keep].copy()
            bdf.to_csv(os.path.join(DATA, "capital_quarterly", f"{stock}.csv"), index=False)
            if len(bdf) > 0:
                latest = bdf.sort_values("REPORT_DATE").iloc[-1]
                records.append(dict(
                    bond_code=code, stock_code=stock, name=name,
                    n_resets=row.get("n_resets"), issue_scale=issue_scale,
                    monetary_funds=latest.get("MONETARYFUNDS"),
                    short_loan=latest.get("SHORT_LOAN"),
                    total_liab=latest.get("TOTAL_LIABILITIES"),
                    report_date=latest.get("REPORT_DATE"),
                ))
        except Exception as e:
            print(f"[capital] {stock} {name} FAILED: {e}")

        time.sleep(1.5)
        print(f"[{i+1}/{len(sample)}] done {code} {name}", flush=True)

    if records:
        pd.DataFrame(records).to_csv(
            os.path.join(DATA, "capital_summary.csv"), index=False)
    print("[done] data collection complete")


if __name__ == "__main__":
    main()
