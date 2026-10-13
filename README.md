# Endogenous Downward Revision in Chinese Convertible Bonds

Pricing Chinese convertible bonds with an **endogenous downward-revision right**,
coupling the issuer's operating capital to the underlying stock price so that a
liquidity-constrained firm faces a self-reinforcing *death spiral* the revision
clause is designed to break.

## Model

The stock price $S_t$ and the operating capital $C_t$ evolve as a bidirectionally
coupled system:

```
dS = [r - q + mu(C)] S dt + sigma_s(C) S dW^s
dC = theta*(C_bar - C) dt + alpha * dS/S + sigma_c C dW^c - dDischarge
```

with a distress drift `mu(C) = -eta * max(0, (C_safe - C)/C_safe)` and a
liquidity-amplified volatility `sigma_s(C) = sigma0 * (1 + beta/max(C, eps))`.
The downward-revision decision is **endogenous**: the issuer revises the
conversion price when the 15-of-30 price trigger is active **and** the capital
buffer falls below a strategic threshold `C_reset`. The resulting game option is
solved by least-squares Monte Carlo (LSM).

## Key results

| specification | bond value | mean revisions/path |
|---|---|---|
| no revision | 114.38 | 0.000 |
| endogenous revision | 115.88 | 0.158 |
| exogenous revision (rho=0.5) | 129.22 | 1.936 |

The death spiral is concentrated among liquidity-constrained firms: a firm with
stock at 70 and depleted capital (C0=50) revises 1.8 times per path and is worth
117.9, versus 99.3 for a cash-rich firm (C0=500) that never revises.
Calibration to 22 Chinese issuers that exercised the revision shows that about a
third sit below the calibrated capital threshold.

## Repository layout

```
erccb/                 pricing engine (numba-accelerated)
  params.py            contract + coupling parameters
  simulate.py          forward simulation of the coupled SDE
  lsm.py               least-squares Monte Carlo backward induction
  engine.py            high-level pricing API
  calibration.py       stock/capital parameter estimation
experiments/
  collect_data.py      fetch market data (akshare)
  backfill_stock.py    stock-daily fallback via Tencent
  run_experiments.py   E1-E4 experiments
  run_calibration.py   parameter calibration
  run_empirical_summary.py  capital-coverage cross-section
  make_figures.py      figures
data/                  collected market data
output/data/           experiment and calibration outputs
output/figures/        figures
```

## Reproduce

```bash
pip install numpy numba scipy pandas akshare matplotlib
export PYTHONPATH=/path/to/EReset_CCB
python experiments/run_experiments.py       # E1-E4
python experiments/run_calibration.py       # parameter calibration
python experiments/run_empirical_summary.py # cross-section
python experiments/make_figures.py          # figures
```

Data collection requires network access to Eastmoney / jisilu; the collected
snapshots are bundled under `data/` so the experiments run offline.

## Data sources

Market data come from the public databases of Eastmoney (东方财富) and jisilu
(集思录), retrieved through the `akshare` Python interface. The underlying
market data are published by Eastmoney and jisilu and are not the authors' own
data; cite those sources when reusing them.

## Citation

Xiaoping Hu. "Endogenous Downward Revision and the Liquidity Death Spiral:
Pricing Chinese Convertible Bonds with Bidirectional Capital-Price Coupling."
Working paper, Southeast University.
