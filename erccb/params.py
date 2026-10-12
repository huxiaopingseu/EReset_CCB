"""Model and contract parameters for the ER-CCB game-option model."""
from dataclasses import dataclass, field, asdict
from typing import List


# Contract clauses (15/30 rule) and monitoring constants.
CONTRACT_KEYS = dict(
    WINDOW=30,          # rolling window of trading days
    HITS=15,            # number of hits needed inside the window
    CALL_MULT=1.30,     # soft-call trigger: S >= 1.30 K
    PUT_MULT=0.70,      # soft-put trigger: S <= 0.70 K
    RESET_MULT=0.85,    # reset trigger:  S <= 0.85 K
    CALL_FROM=0.5,      # call active from year 0.5
    PUT_FROM=5.0,       # put active in years 5 and 6
)


@dataclass
class ERCCBParams:
    # ---- contract -------------------------------------------------
    T: float = 6.0              # maturity (years)
    n_days: int = 252           # trading days per year
    F: float = 100.0            # face value
    S0: float = 100.0           # initial stock price
    K0: float = 100.0           # initial conversion price
    B: float = 5.0              # net asset value per share (reset floor)
    coupons: List[float] = field(
        default_factory=lambda: [0.004, 0.006, 0.010, 0.015, 0.018, 0.020]
    )

    # ---- market / stock ------------------------------------------
    r: float = 0.03             # risk-free rate
    q: float = 0.0              # dividend yield

    # ---- coupled capital-stock dynamics --------------------------
    sigma0: float = 0.30        # base stock volatility (healthy firm)
    eta: float = 0.15           # distress-drift penalty slope
    beta: float = 0.05          # liquidity-amplified volatility coefficient
    C_safe: float = 200.0       # safety threshold of operating capital
    theta: float = 0.80         # mean-reversion speed of capital
    C_bar: float = 300.0        # long-run (target) capital level
    alpha: float = 0.50         # capital feedback to stock returns
    sigma_c: float = 0.25       # capital volatility
    rho: float = 0.60           # instantaneous corr(Ws, Wc)

    # ---- endogenous reset policy ---------------------------------
    C_reset: float = 150.0      # capital threshold: reset iff trigger AND C<C_reset

    # ---- initial capital / distress ------------------------------
    C0: float = 300.0           # initial operating capital
    recovery: float = 0.60      # recovery rate on default

    # ---- numerical -----------------------------------------------
    eps: float = 1e-3           # floor for max(C, eps)

    @property
    def M(self) -> int:
        """Number of daily monitoring steps."""
        return int(round(self.T * self.n_days))

    @property
    def dt(self) -> float:
        return self.T / self.M

    def to_scalars(self):
        """Return a plain tuple of scalars for numba."""
        return (
            self.T, self.n_days, self.F, self.S0, self.K0, self.B,
            tuple(self.coupons),
            self.r, self.q,
            self.sigma0, self.eta, self.beta, self.C_safe,
            self.theta, self.C_bar, self.alpha, self.sigma_c, self.rho,
            self.C_reset, self.C0, self.recovery, self.eps,
        )


def default_params() -> ERCCBParams:
    return ERCCBParams()
