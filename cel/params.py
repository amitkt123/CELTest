"""
CEL model parameters.

Field names, types and units live here. Values live in
calibration/parameters.csv, which also records range, prior, class,
source and grade for every parameter (spec §10.6). S0 is loaded from that
file, so every number in the model traces to a row with a source.

Scenarios are expressed as `S0.with_(**overrides)` so a scenario is
literally a diff against baseline and nothing can drift.

Units: money is constant 2024 USD; compute in FLOP (stocks in FLOP/s);
power in kW; time in years. One unit of task output is one worker-year
at gamma = 1. Labor quantities are per worker: L_bar = 1 is one average
worker, and L_workers converts per-worker quantities to global totals.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace, asdict, fields
from pathlib import Path
from typing import Dict

CSV_PATH = Path(__file__).resolve().parent.parent / "calibration" / "parameters.csv"


@dataclass(frozen=True)
class Params:
    # ------------------------------------------------------------------ time
    t0_year: int                     # first period
    T: int                           # periods; t0_year + T = 2060

    # ------------------------------------------------ Block 1: hardware capital
    eps0: float                      # FLOP/s per kW at t0, all-in
    eps_headroom: float              # eps_max / eps0
    g_eps: float                     # initial efficiency growth /yr
    kappa0: float                    # $ per (FLOP/s) installed at t0
    g_kappa0: float                  # calibration check: ~ g_eps + g_kappa_fab
    g_kappa_fab: float               # residual fab learning /yr
    m_H: float                       # hardware vendor markup, price / cost
    delta_phys: float                # physical retirement /yr
    rho: float                       # private cost of capital
    util: float                      # utilization
    unit_kW: float                   # power per accelerator unit, all-in
    s_T: float                       # training share of compute

    # ------------------------------------------------ Block 2: power / energy
    P_bar0_GW: float                 # AI-attributable power at t0
    g_P0: float                      # near-term buildout rate (decays)
    g_P_inf: float                   # long-run grid growth
    tau_P: float                     # decay time of buildout rate (yr)
    PUE: float
    p_E: float                       # $/kWh
    pi_E: float                      # drift in electricity price /yr
    eta_E: float                     # extra drift when power constraint binds
    iota0: float                     # tCO2 per kWh
    g_iota: float                    # grid decarbonization /yr
    SCC: float                       # $/tCO2

    # ------------------------------------------------ Block 3: capability
    Theta: float                     # training campaign length (yr)
    g_Omega0: float                  # algorithmic progress /yr, initial
    tau_Omega: float                 # decay time of algorithmic progress (yr)
    g_Omega_inf: float               # inference-side efficiency /yr, initial
    x_t0: float                      # calibration target: log10 frontier compute at t0
    x50: float                       # log10 compute for 50% coverage
    s_diff: float                    # difficulty spread (decades per logit)
    phi_max: float                   # automatable ceiling
    e0: float                        # inference FLOP per worker-year at median difficulty
    theta: float                     # inference-cost elasticity to difficulty (per decade)

    # ------------------------------------------------ composition / oversight
    psi: float                       # Clayton dependence across steps
    lam: float                       # graded propagation coefficient
    rho_conceal: float               # self-concealing error share
    q_bar: float                     # detection prob. of non-concealed errors
    omega0: float                    # calibration target: oversight labor share at t0

    # ------------------------------------------------ Block 4: production / labor
    sigma: float                     # task elasticity of substitution
    nu: float                        # new-task creation /yr
    A_Y: float                       # TFP scale, $ per worker-year at K_o = 1
    N0: float                        # initial task measure
    L_bar: float                     # labor per worker (normalized to 1)
    eps_L: float                     # labor supply elasticity
    w_res_ratio: float               # reservation / initial wage (statutory floor)

    # ------------------------------------------------ Block 5: industry
    N_F: int                         # effective competitors
    F0_trillion: float               # financing cap at 2026, $T/yr

    # ------------------------------------------------ displacement / queue
    A_max: int                       # queue-age cohorts tracked (yr)
    mu0: float                       # initial retraining capacity, share of L_bar / yr
    tau_c: int                       # capacity build lag (yr)
    chi: float                       # capacity adjustment cost curvature
    xi_reemp: float                  # re-employment hazard decay with queue age
    pi_scar0: float                  # base permanent-exit hazard /yr
    pi_scar_slope: float             # increase in exit hazard per year queued
    turnover_ceiling: float          # cohort-replacement absorption cap, share/yr
    c2_burst: float                  # Kingman burstiness (c_a^2 + c_s^2)/2

    # ------------------------------------------------ Block 6: welfare
    beta: float
    eta_ia: float
    g0: float                        # no-AI growth
    Y0_trillion: float

    # ------------------------------------------------ task grid
    M_tasks: int                     # power of two for Sobol
    sobol_seed: int
    kappaE_mu: float                 # log error-consequence, relative to task value
    kappaE_sig: float
    Gamma_mu: float                  # log modularity cost
    Gamma_sig: float

    # ----------------------------------------------------------- helpers
    def with_(self, **kw) -> "Params":
        unknown = set(kw) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
        return replace(self, **kw)

    def to_dict(self) -> Dict:
        return asdict(self)


def load_params(path: Path = CSV_PATH) -> Params:
    """Build Params from the provenance CSV. Field set and CSV rows must match exactly."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = {row["name"]: row for row in csv.DictReader(fh)}
    names = {f.name for f in fields(Params)}
    missing = sorted(names - set(rows))
    extra = sorted(set(rows) - names)
    if missing or extra:
        raise KeyError(f"parameters.csv out of sync: missing={missing} extra={extra}")
    kw = {}
    for f in fields(Params):
        raw = rows[f.name]["value"]
        kw[f.name] = int(raw) if f.type == "int" else float(raw)
    return Params(**kw)


S0 = load_params()

# Scenario diffs (spec §11.2). Each is a dict so the diff is auditable.
SCENARIOS: Dict[str, Dict] = {
    "S0_baseline": {},
    "S1_motivating": dict(phi_max=0.98, nu=0.0, sigma=1.2, eps_headroom=10.0, tau_Omega=4.0),
    "S2_scaling":    dict(eps_headroom=300.0, tau_Omega=20.0, g_P_inf=0.08),
    "S3_baumol":     dict(sigma=0.4, nu=0.01),
    "S4_powerbound": dict(g_P_inf=0.02, g_P0=0.25),
}


def scenario(name: str) -> Params:
    return S0.with_(**SCENARIOS[name])
