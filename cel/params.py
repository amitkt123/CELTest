"""
CEL model parameters.

One frozen dataclass holds every parameter with its S0 baseline value.
Scenarios are expressed as `dataclasses.replace(S0, **overrides)` so a
scenario is literally a diff against baseline and nothing can drift.

Units are stated per field. Money is constant 2024 USD. Compute is in
effective FLOP (stock quantities in FLOP/s). Power in kW. Time in years.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict, fields
from typing import Dict


@dataclass(frozen=True)
class Params:
    # ------------------------------------------------------------------ time
    t0_year: int = 2023
    T: int = 37                      # periods; t0_year + T = 2060

    # ------------------------------------------------ Block 1: hardware capital
    eps0: float = 1.4e15             # FLOP/s per kW at t0 (H100-class, all-in)
    eps_headroom: float = 30.0       # eps_max / eps0  [10, 300]  (bold)
    g_eps: float = 0.28              # initial efficiency growth /yr
    kappa0: float = 3.0e-11          # $ per (FLOP/s) installed at t0 (~$30k / 1e15)
    g_kappa0: float = 0.30           # quality-adjusted price decline /yr
    g_kappa_fab: float = 0.03        # residual fab learning /yr
    m_H: float = 3.0                 # hardware vendor markup (75% gross margin)
    delta_phys: float = 0.15         # physical retirement /yr
    rho: float = 0.10                # cost of capital
    util: float = 0.5                # utilization
    unit_kW: float = 1.0             # power per accelerator unit, all-in
    s_T: float = 0.30                # training share of compute

    # ------------------------------------------------ Block 2: power / energy
    P_bar0_GW: float = 15.0          # AI-attributable power at t0 (2023); ~50 GW by 2026
    g_P0: float = 0.45               # near-term buildout rate (decays)
    g_P_inf: float = 0.04            # long-run grid growth
    tau_P: float = 6.0               # decay time of buildout rate (yr)
    PUE: float = 1.2
    p_E: float = 0.07                # $/kWh
    pi_E: float = 0.01               # drift in electricity price /yr
    eta_E: float = 0.03              # extra drift when power constraint binds
    iota0: float = 0.38e-3           # tCO2 per kWh
    g_iota: float = 0.03             # grid decarbonization /yr
    SCC: float = 190.0               # $/tCO2

    # ------------------------------------------------ Block 3: capability
    Theta: float = 0.25              # training campaign length (yr)
    g_Omega0: float = 0.60           # algorithmic progress /yr           (bold)
    tau_Omega: float = 8.0           # its decay time (yr)                (bold)
    g_Omega_inf: float = 0.80        # inference-side efficiency /yr
    x_t0: float = 26.0               # log10 effective training compute at t0
    x50: float = 29.0                # log10 compute for 50% coverage     (bold)
    s_diff: float = 1.5              # difficulty spread (decades/logit)  (bold)
    phi_max: float = 0.75            # automatable ceiling                (bold)
    e0: float = 1.0e15               # inference FLOP per task at MEDIAN difficulty (d = x50)
    theta: float = 1.0               # inference-cost elasticity to difficulty (per decade)

    # ------------------------------------------------ composition / oversight
    psi: float = 0.5                 # Clayton dependence across steps
    lam: float = 0.10                # graded propagation coefficient
    rho_conceal: float = 0.15        # self-concealing error share
    q_bar: float = 0.90              # detection prob. of non-concealed errors
    omega0: float = 0.30             # initial oversight labor fraction (hook only)

    # ------------------------------------------------ Block 4: production / labor
    sigma: float = 0.60              # task elasticity of substitution   (bold; threshold 1)
    nu: float = 0.005                # new-task creation /yr              (bold)
    A_Y: float = 1.0                 # TFP scale
    N0: float = 1.0                  # initial task measure
    L_bar: float = 1.0               # workforce (normalized; 3.5e9 real)
    eps_L: float = 0.30              # labor supply elasticity
    w_res_ratio: float = 0.50        # reservation / initial wage (statutory floor)

    # ------------------------------------------------ Block 5: industry
    N_F: int = 5                     # effective competitors
    F0_trillion: float = 0.6         # financing cap at 2026, $T/yr

    # ------------------------------------------------ displacement / queue
    A_max: int = 20                  # queue-age cohorts tracked (yr)
    mu0: float = 0.02                # initial retraining capacity, share of L_bar / yr
    tau_c: int = 4                   # capacity build lag (yr)
    chi: float = 2.0                 # capacity adjustment cost curvature
    xi_reemp: float = 0.35           # re-employment hazard decay with queue age
    pi_scar0: float = 0.03           # base permanent-exit hazard /yr
    pi_scar_slope: float = 0.02      # increase in exit hazard per year queued
    turnover_ceiling: float = 0.025  # cohort-replacement absorption cap, share/yr
    c2_burst: float = 1.0            # Kingman burstiness (c_a^2 + c_s^2)/2; cascades raise it

    # ------------------------------------------------ Block 6: welfare
    beta: float = 0.97
    eta_ia: float = 1.5
    g0: float = 0.025                # no-AI growth
    Y0_trillion: float = 105.0

    # ------------------------------------------------ task grid
    M_tasks: int = 4096              # power of two for Sobol
    sobol_seed: int = 7
    kappaE_mu: float = 0.0           # log error-consequence, relative to task value
    kappaE_sig: float = 1.0
    Gamma_mu: float = 0.0            # log modularity cost
    Gamma_sig: float = 0.8

    # ----------------------------------------------------------- helpers
    def with_(self, **kw) -> "Params":
        unknown = set(kw) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
        return replace(self, **kw)

    def to_dict(self) -> Dict:
        return asdict(self)


S0 = Params()

# Scenario diffs (spec §11). Each is a dict so the diff is auditable.
SCENARIOS: Dict[str, Dict] = {
    "S0_baseline": {},
    "S1_motivating": dict(phi_max=0.98, nu=0.0, sigma=1.2, eps_headroom=10.0, tau_Omega=4.0),
    "S2_scaling":    dict(eps_headroom=300.0, tau_Omega=20.0, g_P_inf=0.08),
    "S3_baumol":     dict(sigma=0.4, nu=0.01),
    "S4_powerbound": dict(g_P_inf=0.02, g_P0=0.25),
}


def scenario(name: str) -> Params:
    return S0.with_(**SCENARIOS[name])
