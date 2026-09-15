"""
Displaced-worker pool as a cohort array indexed by queue age (years).

S[a] = mass of workers who have been displaced for a years.

Each period:
  1. exits    : permanent labor-force exit with hazard pi(a) = pi0 + slope*a
  2. outflow  : re-employment, capped at min(capacity mu, matching M);
                allocated across ages by a hazard proportional to
                exp(-xi * a)  (employers prefer short-duration candidates)
  3. ageing   : survivors move to a+1; new displaced enter at a = 0
  4. duration : Little's law  varrho = S.sum() / outflow  (endogenous)

This is deliberately minimal: the point of the cohort structure is that
varrho and scarring depend on the *distribution* of durations, which a
scalar pool cannot represent.
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
        # 2. re-employment with Kingman-type congestion
        #    A deterministic fluid queue has NO congestion below capacity
        #    (duration flat, then a kink). The (mu - lambda)^{-1} explosion
        #    is a stochastic result. We import it via Kingman's approximation:
        #    expected wait ~ u/(1-u) * c2 / h, u = arrival utilization,
        #    c2 = (c_a^2 + c_s^2)/2 arrival+service burstiness. Cascades raise c_a^2.
        cap = capacity if match_cap is None else min(capacity, match_cap)
        cap = max(cap, 0.0)
        haz = np.exp(-p.xi_reemp * self.ages)
        want = S * haz
        total_want = want.sum()
        if total_want > 0.0 and cap > 0.0:
            u = inflow / cap
            congestion = 1.0 / (1.0 + p.c2_burst * u / (1.0 - u)) if u < 1.0 else 1.0
            scale = min(congestion, cap / total_want)   # throughput never exceeds cap
            out_by_age = want * scale
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
