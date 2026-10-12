"""Endogenous-Reset Convertible Bond (ER-CCB) pricing package.

Models a Chinese convertible bond with a *downward-reset* clause whose
triggering is ENDOGENOUS: the issuer resets the conversion price only when
the 15/30 price trigger is active AND the firm's operating-capital buffer
falls below a strategic threshold.  The stock price and the operating
capital are bidirectionally coupled through a distress-drift and a
liquidity-amplified volatility.
"""
import os

# Avoid the numba/TBB + numpy 2.4 heap-corruption segfault under parallel=True.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 4))

from .params import ERCCBParams, CONTRACT_KEYS, default_params
from .simulate import simulate_paths
from .lsm import lsm_price
from .engine import price_ccb, price_benchmarks

__all__ = [
    "ERCCBParams", "CONTRACT_KEYS", "default_params",
    "simulate_paths", "lsm_price", "price_ccb", "price_benchmarks",
]
