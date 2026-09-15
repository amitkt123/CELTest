"""
Task grid.

Tasks are a quasi-Monte-Carlo (Sobol) sample from the joint distribution
F(d, kappa_E, Gamma, a). Because the Sobol points are uniform in CDF space
and mapped through quantile functions, each point carries equal measure
weight N0 / M. That is the F-weighting; no importance weights are needed.

Ordering is fixed once at construction. Never re-sort: flip diagnostics
track tasks by index.

    d       : log10 effective compute at which AI reaches human parity
              (logistic with location x50, scale s_diff -> H in spec §4.3)
    kappa_E : error consequence, relative to task value (lognormal)
    Gamma   : modularity cost of chunking (lognormal)
    a       : automatable flag, Bernoulli(phi_max), independent of d
              (spec §4.3: non-automatability is physical presence,
              liability or preference, not compute difficulty)
    gamma   : human productivity on the task (flat = 1 for now)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc, norm

from .params import Params


class TaskGrid:
    def __init__(self, p: Params):
        m = p.M_tasks
        if m & (m - 1):
            raise ValueError("M_tasks must be a power of two for Sobol balance")
        sob = qmc.Sobol(d=4, scramble=True, seed=p.sobol_seed)
        u = sob.random_base2(int(np.log2(m)))
        u = np.clip(u, 1e-9, 1 - 1e-9)

        self.M = m
        self.d = p.x50 + p.s_diff * np.log(u[:, 0] / (1.0 - u[:, 0]))
        self.kappaE = np.exp(p.kappaE_mu + p.kappaE_sig * norm.ppf(u[:, 1]))
        self.Gamma = np.exp(p.Gamma_mu + p.Gamma_sig * norm.ppf(u[:, 2]))
        self.a = u[:, 3] < p.phi_max
        self.gamma = np.ones(m)
        self.weight = np.full(m, p.N0 / m)     # task measure per point
        # fractional rank by difficulty, in [0,1): diagnostics and tests
        order = np.argsort(self.d, kind="stable")
        self.rank = np.empty(m)
        self.rank[order] = (np.arange(m) + 0.5) / m

    # total task measure N(t); grows with new-task creation later
    @property
    def N(self) -> float:
        return float(self.weight.sum())

    def coverage_at(self, x: float) -> float:
        """Empirical H(x): measure share of tasks with d <= x."""
        return float(self.weight[self.d <= x].sum() / self.N)

    def capable_at(self, x: float) -> np.ndarray:
        """capable_i = [d_i <= x] and [a_i = 1]  (spec §4.3)."""
        return (self.d <= x) & self.a

    def inference_flop(self, p: Params, Omega_inf: float = 1.0) -> np.ndarray:
        """e(i): inference FLOP per worker-year of task output (spec §4.4).

        Anchored at the median: e = e0 * 10**(theta * (d - x50)), so e0 is
        compute per worker-year at median difficulty.
        """
        return p.e0 / Omega_inf * 10.0 ** (p.theta * (self.d - p.x50))
