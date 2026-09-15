"""
Period economy (spec §5.1-5.3), static version.

Units: money is 2024 USD; labor quantities are per worker (L = 1 is one
average worker); one unit of task output is one worker-year at gamma = 1.
r is $ per delivered FLOP, so c_AI(i) = r * e(i) is $ per worker-year.

Production (spec §5.1):
    Y   = A_Y * K_o^alpha * T^(1 - alpha)
    P_T = [ sum_i w_i p_i^(1 - sigma) ]^(1 / (1 - sigma))
    y_i = (p_i / P_T)^(-sigma) * T
    sum_i w_i p_i y_i = P_T * T                     (checked: nominal_check)

Equilibrium in w with P_Y = 1 as numeraire:
    labor clearing    T = L / ell_T(w),  ell_T = sum_human w_i (p_i/P_T)^-sigma / gamma_i
    marginal product  P_T(w) = (1 - alpha) * A_Y * K_o^alpha * T^(-alpha)
The residual log(P_T / MPT) is increasing in w and is solved with brentq
in log w. With alpha = 0 this is exactly the old condition P_T = A_Y.

Automation rule (spec §5.2): task i is done by AI iff
    capable_i  AND  c_AI(i) < w / gamma_i
where capable comes from TaskGrid.capable_at(x). Oversight labor is zero
until build step 6.

Feasibility: with alpha > 0 a root always exists (as w -> inf the human
tasks vanish, T -> inf and the marginal product -> 0). With alpha = 0 and
sigma > 1 the numeraire cannot be met once automated tasks alone push
P_T below A_Y; solve_period raises Infeasible. compute_market treats that
as compute demand exceeding any finite supply.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

from .params import Params
from .grid import TaskGrid

_LN10 = np.log(10.0)


class Infeasible(RuntimeError):
    pass


@dataclass
class PeriodResult:
    w: float                 # wage, $ per worker-year
    r: float                 # compute price, $ per delivered FLOP
    Y: float                 # output, $ per worker
    T_agg: float             # task aggregate
    P_T: float               # task price index
    L_d: float               # labor demand per worker
    K_d: float               # compute demand, delivered FLOP per worker-year
    s_L: float               # labor share
    phi_eff: float           # automated task share
    prices: np.ndarray
    auto: np.ndarray
    nominal_check: float     # sum w_i p_i y_i / (P_T T) - 1, should be ~0
    mpt_check: float         # P_T / marginal product of T - 1, should be ~0


def task_prices(grid: TaskGrid, w: float, c_AI: np.ndarray, capable: np.ndarray,
                force: bool = False):
    """Return (prices, auto_mask). force=True ignores the cost condition (tests)."""
    human_cost = w / grid.gamma
    auto = capable & ((c_AI < human_cost) | force)
    prices = np.where(auto, c_AI, human_cost)
    return prices, auto


def task_price_index(grid: TaskGrid, p: Params, prices: np.ndarray) -> float:
    s = p.sigma
    if abs(s - 1.0) < 1e-9:
        raise ValueError("sigma = 1 (Cobb-Douglas tasks) is not supported; use 1 +/- 1e-3")
    return float(np.sum(grid.weight * prices ** (1.0 - s)) ** (1.0 / (1.0 - s)))


def solve_period(grid: TaskGrid, p: Params, r: float, capable: np.ndarray,
                 K_o: float = 1.0, L_eff: float | None = None,
                 Omega_inf: float = 1.0, force: bool = False) -> PeriodResult:
    e = grid.inference_flop(p, Omega_inf)
    c_AI = r * e
    L = p.L_bar if L_eff is None else L_eff
    a = p.alpha_K
    scale = p.A_Y * K_o ** a

    def state(w: float):
        prices, auto = task_prices(grid, w, c_AI, capable, force)
        P_T = task_price_index(grid, p, prices)
        human = ~auto
        ell_T = float(np.sum(grid.weight[human] * (prices[human] / P_T) ** (-p.sigma)
                             / grid.gamma[human]))
        return prices, auto, P_T, ell_T

    def resid(log_w: float) -> float:
        _, _, P_T, ell_T = state(float(np.exp(log_w)))
        if ell_T <= 0.0:
            raise Infeasible("no human tasks remain; T unbounded under labor clearing")
        mpt = (1.0 - a) * scale * (L / ell_T) ** (-a)
        return float(np.log(P_T / mpt))

    centre = float(np.log(scale))
    hi = centre + 1.0
    while resid(hi) < 0.0:
        hi += _LN10
        if hi > centre + 12.0 * _LN10:
            raise Infeasible("P_Y = 1 has no solution: automated tasks alone price "
                             "the task aggregate below its marginal product "
                             "(alpha = 0, sigma > 1 and r too low).")
    lo = centre - 1.0
    while resid(lo) > 0.0:
        lo -= _LN10
        if lo < centre - 30.0 * _LN10:
            raise Infeasible("P_T exceeds the marginal product even as w -> 0; check A_Y.")

    w = float(np.exp(brentq(resid, lo, hi, xtol=1e-13, rtol=4 * np.finfo(float).eps)))

    prices, auto, P_T, ell_T = state(w)
    T_agg = L / ell_T
    Y = scale * T_agg ** (1.0 - a)
    y = (prices / P_T) ** (-p.sigma) * T_agg
    human = ~auto
    L_d = float(np.sum(grid.weight[human] * y[human] / grid.gamma[human]))
    K_d = float(np.sum(grid.weight[auto] * e[auto] * y[auto]))
    task_spend = float(np.sum(grid.weight * prices * y))
    return PeriodResult(
        w=w, r=r, Y=Y, T_agg=T_agg, P_T=P_T, L_d=L_d, K_d=K_d, s_L=w * L_d / Y,
        phi_eff=float(grid.weight[auto].sum() / grid.N),
        prices=prices, auto=auto,
        nominal_check=task_spend / (P_T * T_agg) - 1.0,
        mpt_check=P_T / ((1.0 - a) * scale * T_agg ** (-a)) - 1.0,
    )


def labor_share_closed_form(p: Params, phi: float, w: float, p_A: float, N: float = 1.0) -> float:
    """Proposition 3, symmetric case: gamma = 1, uniform AI price p_A."""
    s = p.sigma
    inner = (N - phi) * w ** (1 - s) / (phi * p_A ** (1 - s) + (N - phi) * w ** (1 - s))
    return (1.0 - p.alpha_K) * inner
