"""Backfill stock daily data via the Tencent endpoint (Eastmoney is throttling).

For each stock in sample_bonds.csv lacking a stock_daily file, fetch daily
closes from ak.stock_zh_a_hist_tx (source: Tencent).  Writes data/stock_daily/*.csv.
"""
import os
import time
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SDIR = os.path.join(DATA, "stock_daily")
os.makedirs(SDIR, exist_ok=True)


def retry(fn, *args, tries=4, base_delay=2.0, **kw):
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(base_delay * (2 ** i))
    return None


def main():
    sample = pd.read_csv(os.path.join(DATA, "sample_bonds.csv"))
    stocks = sample["正股代码"].astype(int).astype(str).str.zfill(6).tolist()

    done = 0
    for stock in stocks:
        out = os.path.join(SDIR, f"{stock}.csv")
        if os.path.exists(out):
            done += 1
            continue
        # Tencent symbol prefix
        if stock.startswith(("6", "9")):
            sym = "sh" + stock
        else:
            sym = "sz" + stock
        try:
            df = retry(ak.stock_zh_a_hist_tx, symbol=sym,
                       start_date="20190101", end_date="20261231", adjust="")
            df = df[["date", "close"]]
            df.to_csv(out, index=False)
            done += 1
            print(f"[stock-tx] {stock} OK rows={len(df)}", flush=True)
        except Exception as e:
            print(f"[stock-tx] {stock} FAILED: {e}", flush=True)
        time.sleep(1.0)

    print(f"[done] stock backfill: {done}/{len(stocks)} present")


if __name__ == "__main__":
    main()
