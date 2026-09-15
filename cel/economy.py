"""
Period economy (spec Block 4), static version for build steps 1-2.

Given the rental price of compute r (here $ per FLOP, i.e. already
converted from $ per (FLOP/s)-year by dividing by 3.156e7), a capability
coverage phi_cap, and the task grid, find the wage w such that the
final-good price index equals one (the numeraire), then compute output,
factor demands, and shares.

CES accounting used throughout (A_Y general):
    P_Y  = A_Y^{-1} [ sum_i w_i p_i^{1-sigma} ]^{1/(1-sigma)}
    y_i  = (p_i / P_Y)^{-sigma} * Y * A_Y^{sigma-1}
    sum_i w_i p_i y_i = P_Y * Y                    (checked in tests)

Automation rule (spec §5.2): task i is done by AI iff
    rank_i < phi_cap   AND   c_AI(i) < w / gamma_i
so phi_eff <= phi_cap and the gap is the deployment wedge.

Feasibility: with sigma > 1 the normalization P_Y = 1 has no solution if
automated tasks alone already push unit cost below 1, i.e.
    sum_auto w_i c_AI(i)^{1-sigma} > A_Y^{1-sigma}.
That is not a bug; it says the given r is inconsistent with the numeraire.
In the dynamic model r is endogenous and bounded below by energy cost, so
this only arises for silly exogenous test values. We raise Infeasible.

Oversight labor is a hook (omega = 0) until step 6 replaces it with the
chunking optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import Params
from .grid import TaskGrid


class Infeasible(RuntimeError):
    pass


@dataclass
class PeriodResult:
    w: float
    r: float
    Y: float
    L_d: float
    K_d: float
    s_L: float
    phi_cap: float
    phi_eff: float
    prices: np.ndarray
    auto: np.ndarray
    P_Y: float
    nominal_check: float   # sum w_i p_i y_i / (P_Y * Y) - 1, should be ~0


def task_prices(grid: TaskGrid, p: Params, w: float, c_AI: np.ndarray, phi_cap: float,
                force: bool = False):
    """Return (prices, auto_mask). force=True ignores the cost condition (tests)."""
    human_cost = w / grid.gamma
    capable = grid.rank < phi_cap
    economical = c_AI < human_cost
    auto = capable & (economical | force)
    prices = np.where(auto, c_AI, human_cost)
    return prices, auto


def price_index(grid: TaskGrid, p: Params, prices: np.ndarray) -> float:
    s = p.sigma
    agg = np.sum(grid.weight * prices ** (1.0 - s))
    return float(agg ** (1.0 / (1.0 - s)) / p.A_Y)


def solve_period(grid: TaskGrid, p: Params, r: float, phi_cap: float,
                 Omega_inf: float = 1.0, force: bool = False,
                 tol: float = 1e-10, max_iter: int = 200) -> PeriodResult:
    """Bisection in w on the monotone residual P_Y(w) - 1."""
    e = grid.inference_flop(p, Omega_inf)
    c_AI = r * e

    def resid(w: float) -> float:
        prices, _ = task_prices(grid, p, w, c_AI, phi_cap, force)
        return price_index(grid, p, prices) - 1.0

    lo, hi = 1e-12, 1.0
    # expand bracket upward; P_Y is nondecreasing in w
    while resid(hi) < 0.0:
        hi *= 2.0
        if hi > 1e12:
            raise Infeasible("P_Y = 1 has no solution: automated tasks alone "
                             "price the numeraire below 1 (sigma > 1 and r too low).")
    if resid(lo) > 0.0:
        raise Infeasible("P_Y > 1 even at w -> 0; check A_Y.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if resid(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    w = 0.5 * (lo + hi)

    prices, auto = task_prices(grid, p, w, c_AI, phi_cap, force)
    P_Y = price_index(grid, p, prices)
    s = p.sigma
    human = ~auto
    demand_scale = p.A_Y ** (s - 1.0)         # A_Y^{sigma-1} factor in y_i
    # labor market clearing pins Y (oversight term = 0 for now)
    denom = demand_scale * np.sum(grid.weight[human] * prices[human] ** (-s) / grid.gamma[human])
    if denom <= 0.0:
        raise Infeasible("no human tasks remain; Y unbounded under this closure")
    Y = p.L_bar / denom
    y = (prices / P_Y) ** (-s) * Y * demand_scale
    L_d = float(np.sum(grid.weight[human] * y[human] / grid.gamma[human]))
    K_d = float(np.sum(grid.weight[auto] * e[auto] * y[auto]))
    nominal = float(np.sum(grid.weight * prices * y))
    return PeriodResult(
        w=w, r=r, Y=Y, L_d=L_d, K_d=K_d, s_L=w * L_d / Y,
        phi_cap=phi_cap, phi_eff=float(grid.weight[auto].sum() / grid.N),
        prices=prices, auto=auto, P_Y=P_Y,
        nominal_check=nominal / (P_Y * Y) - 1.0,
    )


def labor_share_closed_form(p: Params, phi: float, w: float, p_A: float, N: float = 1.0) -> float:
    """Proposition 3, symmetric case: gamma = 1, uniform AI price p_A."""
    s = p.sigma
    return (N - phi) * w ** (1 - s) / (phi * p_A ** (1 - s) + (N - phi) * w ** (1 - s))
