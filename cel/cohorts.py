"""
Displaced-worker pool as a cohort array indexed by queue age (years).

S[a] = mass of workers who have been displaced for a years.

Each period:
  1. exits    : permanent labor-force exit with hazard pi(a) = pi0 + slope*a
  2. outflow  : re-employment through a clearing function (spec §7.3)
                    X = mu * S_eff / (S_eff + c2 * mu / h0)
                S_eff = sum_a S[a] exp(-xi a) is the employable stock;
                X is allocated across ages in proportion to S[a] exp(-xi a)
                (employers prefer short-duration candidates)
  3. ageing   : survivors move to a+1; new displaced enter at a = 0
  4. duration : Little's law  varrho = S.sum() / outflow  (endogenous)

The clearing function replaces an inflow-ratio congestion factor that was
switched off at saturation, which made duration jump from ~2000 yr at
u = 0.9995 to 1 yr at u = 1. Here outflow depends on the stock, so the
steady state below capacity is W = c2 / (h0 (1 - u)) (Kingman's formula
exactly at c2 = 1, same heavy-traffic limit for any c2) and for u >= 1
the stock grows without bound. W is continuous and monotone in u.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .params import Params


@dataclass
class QueueStep:
    outflow: float
    exits: float
    stock: float
    mean_duration: float     # Little's-law varrho; inf if no outflow
    saturation: float        # inflow / capacity this period


class DisplacedPool:
    def __init__(self, p: Params):
        self.p = p
        self.S = np.zeros(p.A_max)
        self.ages = np.arange(p.A_max, dtype=float)

    def step(self, inflow: float, capacity: float, match_cap: float | None = None) -> QueueStep:
        p = self.p
        S = self.S
        # 1. scarring exits
        pi = np.clip(p.pi_scar0 + p.pi_scar_slope * self.ages, 0.0, 1.0)
        exits_by_age = S * pi
        S = S - exits_by_age
        # 2. re-employment through the clearing function
        cap = capacity if match_cap is None else min(capacity, match_cap)
        cap = max(cap, 0.0)
        employable = S * np.exp(-p.xi_reemp * self.ages)
        S_eff = employable.sum()
        if S_eff > 0.0 and cap > 0.0:
            X = cap * S_eff / (S_eff + p.c2_burst * cap / p.h0_reemp)
            # with c2 < h0 the per-head rate X / S_eff can exceed 1; never
            # take more workers out of a cohort than it holds
            out_by_age = np.minimum(employable * (X / S_eff), S)
        else:
            out_by_age = np.zeros_like(S)
        S = S - out_by_age
        # 3. ageing; last cohort absorbs overflow
        aged = np.zeros_like(S)
        aged[1:] = S[:-1]
        aged[-1] += S[-1]
        aged[0] = max(inflow, 0.0)
        self.S = aged
        out = float(out_by_age.sum())
        return QueueStep(
            outflow=out,
            exits=float(exits_by_age.sum()),
            stock=float(self.S.sum()),
            mean_duration=float(self.S.sum() / out) if out > 0 else float("inf"),
            saturation=float(inflow / capacity) if capacity > 0 else float("inf"),
        )
