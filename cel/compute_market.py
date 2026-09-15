"""
Within-period compute market (spec §9.2, build step 3).

Supply is a merit order: vintage v offers capacity cap_v (delivered FLOP
per worker-year) at operating cost op_v ($ per delivered FLOP). Demand is
K_d(r) from the period economy, which is nonincreasing in r.

The clearing price is the smallest r with K_d(r) <= S(r). Both curves are
step functions (tasks flip between AI and human; vintages switch on), so
the root is bracketed by bisection in log r rather than brentq.

Regimes (marginal vintage = dearest vintage dispatched at r):
    slack  : r equals the marginal vintage's operating cost
    scarce : r exceeds it; every dispatched vintage runs at capacity and
             the premium r - op_marginal is a scarcity rent (to power
             owners if power binds, otherwise to compute owners; spec §3.2).
             Dearer vintages may sit idle.

When a block of tasks flips at a single price (symmetric tasks, theta = 0)
K_d jumps across capacity and the solver returns the upper side of the
jump with utilization < 1. With heterogeneous tasks the jump is one grid
point and utilization is ~1.

solve_period raising Infeasible (alpha = 0, sigma > 1, r very low) means
demand exceeds any finite supply, so it is treated as positive excess
demand and the price rises. A full run therefore never raises Infeasible.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import Params
from .grid import TaskGrid
from .economy import PeriodResult, Infeasible, solve_period


class MarketError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComputeSupply:
    capacity: np.ndarray     # delivered FLOP per worker-year, per vintage
    op_cost: np.ndarray      # $ per delivered FLOP, per vintage

    def __post_init__(self):
        cap = np.asarray(self.capacity, dtype=float)
        op = np.asarray(self.op_cost, dtype=float)
        if cap.shape != op.shape or cap.ndim != 1 or cap.size == 0:
            raise ValueError("capacity and op_cost must be equal-length 1-D arrays")
        if np.any(cap < 0.0) or np.any(op <= 0.0):
            raise ValueError("capacity must be >= 0 and op_cost > 0")
        object.__setattr__(self, "capacity", cap)
        object.__setattr__(self, "op_cost", op)

    @property
    def total(self) -> float:
        return float(self.capacity.sum())

    def at(self, r: float) -> float:
        """Capacity dispatched at price r: every vintage with op_cost <= r."""
        return float(self.capacity[self.op_cost <= r].sum())


@dataclass
class MarketResult:
    period: PeriodResult
    r: float
    regime: str              # "slack" or "scarce"
    utilization: float       # K_d / total capacity
    scarcity_rent: float     # r - op_marginal if scarce else 0, $ per delivered FLOP


def solve_market(grid: TaskGrid, p: Params, supply: ComputeSupply, capable: np.ndarray,
                 K_o: float = 1.0, L_eff: float | None = None, Omega_inf: float = 1.0,
                 rtol: float = 1e-12, max_expand: int = 200) -> MarketResult:
    def excess(r: float):
        try:
            res = solve_period(grid, p, r, capable, K_o=K_o, L_eff=L_eff, Omega_inf=Omega_inf)
        except Infeasible:
            return float("inf"), None
        return res.K_d - supply.at(r), res

    lo = float(supply.op_cost.min())
    ex, res = excess(lo)
    if ex <= 0.0:
        return _result(res, lo, supply)

    hi = lo
    for _ in range(max_expand):
        hi *= 2.0
        ex, res = excess(hi)
        if ex <= 0.0:
            break
    else:
        raise MarketError("compute demand exceeds supply at every price tried")

    # invariant: excess(lo) > 0 >= excess(hi)
    while hi / lo - 1.0 > rtol:
        mid = float(np.sqrt(lo * hi))
        ex_mid, res_mid = excess(mid)
        if ex_mid <= 0.0:
            hi, res = mid, res_mid
        else:
            lo = mid
    return _result(res, hi, supply)


def _result(res: PeriodResult, r: float, supply: ComputeSupply) -> MarketResult:
    # the marginal vintage is the dearest one dispatched at r; r >= min op_cost
    # always holds at the solution, so at least one vintage is dispatched
    marginal = float(supply.op_cost[supply.op_cost <= r].max())
    scarce = r > marginal * (1.0 + 1e-9)
    return MarketResult(
        period=res,
        r=r,
        regime="scarce" if scarce else "slack",
        utilization=res.K_d / supply.total if supply.total > 0 else float("inf"),
        scarcity_rent=r - marginal if scarce else 0.0,
    )
