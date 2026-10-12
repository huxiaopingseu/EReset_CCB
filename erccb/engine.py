"""High-level pricing API for the ER-CCB game-option model."""
import numpy as np

from .simulate import simulate_paths
from .lsm import lsm_price


def price_ccb(params, N=20000, seed=0, reset_mode=1, rho_reset=0.5,
              n_replicates=1):
    """Price the ER-CCB under a given reset mode.

    reset_mode: 1 = endogenous (this paper), 2 = exogenous (fixed prob),
                0 = no reset (baseline).
    Returns (price, standard_error) averaged over independent replicates.
    """
    prices = []
    for rep in range(n_replicates):
        sim = simulate_paths(params, N, seed=seed + rep,
                             reset_mode=reset_mode, rho_reset=rho_reset)
        px, se = lsm_price(sim, params)
        prices.append(px)
    price = float(np.mean(prices))
    if n_replicates > 1:
        se = float(np.std(prices, ddof=1) / np.sqrt(n_replicates))
    else:
        # re-run a quick se estimate using the last sim's per-path spread
        sim = simulate_paths(params, N, seed=seed + n_replicates,
                             reset_mode=reset_mode, rho_reset=rho_reset)
        _, se = lsm_price(sim, params)
    return price, se


def price_benchmarks(params, N=20000, seed=0, rho_exo=0.5):
    """Price the three model variants for a fair comparison.

    Returns a dict with keys 'no_reset', 'exogenous', 'endogenous'.
    """
    out = {}
    for key, mode, rho in [
        ("no_reset", 0, 0.0),
        ("exogenous", 2, rho_exo),
        ("endogenous", 1, 0.0),
    ]:
        px, se = price_ccb(params, N=N, seed=seed, reset_mode=mode, rho_reset=rho)
        out[key] = (px, se)
    return out
