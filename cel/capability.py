"""
Algorithmic progress paths (spec §4.1).

Training and inference efficiency multipliers grow at decaying rates and
share the exhaustion timescale tau_Omega:

    ln Omega(tau)     = g_Omega0    * tau_Omega * (1 - exp(-tau / tau_Omega))
    ln Omega_inf(tau) = g_Omega_inf * tau_Omega * (1 - exp(-tau / tau_Omega))

tau is years since t0. The initial growth rate is g and the cumulative
limit is exp(g * tau_Omega). A constant 0.80/yr inference rate would
compound to ~1e13 by 2060; the decay removes that.
"""
from __future__ import annotations

import math

from .params import Params


def _log_multiplier(g: float, tau_decay: float, tau: float) -> float:
    return g * tau_decay * (1.0 - math.exp(-tau / tau_decay))


def omega_train(p: Params, tau: float) -> float:
    """Training-side effective-compute multiplier, Omega(tau), Omega(0) = 1."""
    return math.exp(_log_multiplier(p.g_Omega0, p.tau_Omega, tau))


def omega_inf(p: Params, tau: float) -> float:
    """Inference-side efficiency multiplier, Omega_inf(tau), Omega_inf(0) = 1."""
    return math.exp(_log_multiplier(p.g_Omega_inf, p.tau_Omega, tau))
