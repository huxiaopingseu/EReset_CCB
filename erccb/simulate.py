"""Forward simulation of the bidirectionally coupled stock-capital system.

The stock price S_t and the operating capital C_t evolve as

    dS_t = [r - q + mu(C_t)] S_t dt + sigma_s(C_t) S_t dW^s_t,
    dC_t = [theta*(C_bar - C_t) + alpha*(r - q + mu(C_t))] dt
           + sigma_c C_t dW^c_t - dDischarge_t,

with a distress drift and a liquidity-amplified volatility

    mu(C)     = -eta * max(0, (C_safe - C)/C_safe),
    sigma_s(C)= sigma0 * (1 + beta/max(C, eps)),

and E[dW^s dW^c] = rho dt.  The discharge consists of the annual coupons.
The downward-reset clause is ENDOGENOUS: the issuer resets the conversion
price when the 15/30 price trigger is active and the capital buffer is below
C_reset.  A forced soft call terminates the path; C_t < 0 triggers default.

All quantities are per 100 of face value: coupons are c_k * F, the put
payout is approximately F, and C_t is measured in the same "yuan per 100
face" units so that C_t < F signals an inability to absorb a put.
"""
import numpy as np
from numba import njit, prange

from .params import CONTRACT_KEYS

WINDOW = CONTRACT_KEYS["WINDOW"]
HITS = CONTRACT_KEYS["HITS"]


@njit(cache=True, inline="always")
def _splitmix64(state):
    state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z, state


@njit(cache=True, inline="always")
def _stdnormal(state):
    """Box-Muller standard normal from a splitmix64 state; returns (z, state)."""
    u1_state = state
    z1, state = _splitmix64(u1_state)
    z2, state = _splitmix64(state)
    u1 = (np.float64(z1 >> np.uint64(11)) + 1.0) / 9007199254740992.0
    u2 = np.float64(z2 >> np.uint64(11)) / 9007199254740992.0
    u1 = max(u1, 1e-12)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    return r * np.cos(theta), state


@njit(cache=True, inline="always")
def _uniform(state):
    """Uniform(0,1) from a splitmix64 state; returns (u, state)."""
    z, state = _splitmix64(state)
    return np.float64(z >> np.uint64(11)) / 9007199254740992.0, state


@njit(cache=True, parallel=True)
def _simulate(
    N, M, T, dt, F, S0, K0, B,
    coupons, r, q, sigma0, eta, beta, C_safe, theta, C_bar, alpha,
    sigma_c, rho, C_reset, C0, recovery, eps,
    call_mult, put_mult, reset_mult, call_from, put_from,
    reset_mode, rho_reset, seed,
):
    # day indices for coupon dates (years 1..5)
    coupon_day = np.zeros(6, dtype=np.int32)
    for k in range(1, 6):
        coupon_day[k] = int(round(k * 252))

    S = np.empty((N, M + 1), dtype=np.float64)
    C = np.empty((N, M + 1), dtype=np.float64)
    K = np.empty((N, M + 1), dtype=np.float64)
    put_active = np.zeros((N, M + 1), dtype=np.uint8)
    alive = np.zeros((N, M + 1), dtype=np.uint8)
    call_day = np.full(N, M + 1, dtype=np.int32)
    default_day = np.full(N, M + 1, dtype=np.int32)
    n_resets = np.zeros(N, dtype=np.int32)

    call_from_day = int(round(call_from * 252))
    put_from_day = int(round(put_from * 252))

    for n in prange(N):
        state = np.uint64(seed + n * 1000003)
        state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))

        s = S0
        c = C0
        k = K0
        S[n, 0] = s
        C[n, 0] = c
        K[n, 0] = k
        alive[n, 0] = 1

        # ring buffers for the 15/30 counters of call(0), put(1), reset(2)
        ring = np.zeros((3, WINDOW), dtype=np.uint8)
        cnt = np.zeros(3, dtype=np.int32)

        for t in range(1, M + 1):
            # --- 1. coupled SDE step ---
            mu = -eta * max(0.0, (C_safe - c) / C_safe)
            sig_s = sigma0 * (1.0 + beta / max(c, eps))

            z1, state = _stdnormal(state)
            z2, state = _stdnormal(state)
            dw_s = np.sqrt(dt) * z1
            dw_c = np.sqrt(dt) * (rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z2)

            drift_s = (r - q + mu) * s * dt
            s_new = s + drift_s + sig_s * s * dw_s
            if s_new < 0.0:
                s_new = 0.0

            # deterministic capital feedback from expected equity performance
            feedback = alpha * (r - q + mu) * dt
            c_new = (c + theta * (C_bar - c) * dt + feedback
                     + sigma_c * c * dw_c)
            # discharge: coupon at year-end (years 1..5)
            for kk in range(1, 6):
                if t == coupon_day[kk]:
                    c_new -= coupons[kk] * F

            s = s_new
            c = c_new

            # --- 2. default (insolvency) ---
            if c < 0.0:
                default_day[n] = t
                S[n, t] = s
                C[n, t] = c
                K[n, t] = k
                break

            # --- 3. update 15/30 counters ---
            trig_call = 1 if (t >= call_from_day and s >= call_mult * k) else 0
            trig_put = 1 if (t >= put_from_day and s <= put_mult * k) else 0
            trig_reset = 1 if (s <= reset_mult * k) else 0

            idx = (t - 1) % WINDOW
            for cl in range(3):
                trig = trig_call if cl == 0 else (trig_put if cl == 1 else trig_reset)
                cnt[cl] += trig - ring[cl, idx]
                ring[cl, idx] = trig

            # --- 4. forced soft call (issuer, rigid) ---
            if cnt[0] >= HITS:
                call_day[n] = t
                S[n, t] = s
                C[n, t] = c
                K[n, t] = k
                break

            # --- 5. reset clause ---
            # reset_mode: 0 = none, 1 = endogenous (C < C_reset), 2 = exogenous (fixed prob)
            do_reset = False
            if reset_mode == 1:
                do_reset = (cnt[2] >= HITS) and (c < C_reset)
            elif reset_mode == 2:
                u, state = _uniform(state)
                do_reset = (cnt[2] >= HITS) and (u < rho_reset)

            if do_reset:
                # new conversion price: max(20-day avg, prev close, net asset)
                lo = max(0, t - 20)
                avg20 = 0.0
                for j in range(lo, t):
                    avg20 += S[n, j]
                avg20 /= (t - lo)
                prev_close = S[n, t - 1]
                k_new = avg20
                if prev_close > k_new:
                    k_new = prev_close
                if B > k_new:
                    k_new = B
                k = k_new
                n_resets[n] += 1
                # reset all counters and restart monitoring with the new K
                ring[:, :] = 0
                cnt[:] = 0

            # record put-trigger state for the LSM backward pass
            put_active[n, t] = 1 if (t >= put_from_day and cnt[1] >= HITS) else 0

            S[n, t] = s
            C[n, t] = c
            K[n, t] = k
            alive[n, t] = 1

    return S, C, K, put_active, alive, call_day, default_day, n_resets


def simulate_paths(params, N, seed=0, reset_mode=1, rho_reset=0.5):
    """Simulate N coupled paths under the ER-CCB model.

    reset_mode: 0 = no reset clause, 1 = endogenous reset, 2 = exogenous reset
    (fixed probability rho_reset on trigger).
    """
    p = params
    S, C, K, put_active, alive, call_day, default_day, n_resets = _simulate(
        N, p.M, p.T, p.dt, p.F, p.S0, p.K0, p.B,
        np.asarray(p.coupons, dtype=np.float64),
        p.r, p.q, p.sigma0, p.eta, p.beta, p.C_safe,
        p.theta, p.C_bar, p.alpha, p.sigma_c, p.rho,
        p.C_reset, p.C0, p.recovery, p.eps,
        CONTRACT_KEYS["CALL_MULT"], CONTRACT_KEYS["PUT_MULT"],
        CONTRACT_KEYS["RESET_MULT"], CONTRACT_KEYS["CALL_FROM"],
        CONTRACT_KEYS["PUT_FROM"],
        reset_mode, rho_reset, seed,
    )
    return dict(S=S, C=C, K=K, put_active=put_active, alive=alive,
                call_day=call_day, default_day=default_day, n_resets=n_resets)
