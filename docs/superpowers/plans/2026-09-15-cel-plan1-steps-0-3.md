# CEL Plan 1: Fixes, Capital Nest, Compute Market (build steps 0–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the CEL skeleton into a correct static equilibrium in real dollar units, with an endogenous compute price. This covers spec build steps 0, 1–2 and 3.

**Architecture:** Parameters move to a provenance CSV that `params.py` loads. The period economy gets an outer Cobb–Douglas nest with conventional capital, per-worker dollar units, and a capability mask built from difficulty plus an independent automatability flag. A new compute market wraps the period solve: a merit-order supply curve, and bisection for the market-clearing price. The displaced-worker queue replaces its broken congestion factor with a stock-based clearing function. Algorithmic-progress paths get their own small module.

**Tech Stack:** Python 3.11, numpy, scipy (`qmc.Sobol`, `optimize.brentq`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-cel-model-design.md`

**Roadmap:** this is Plan 1 of 5.

| Plan | Build steps |
|---|---|
| 1 (this one) | 0–3 |
| 2 | 4–5: vintages, power, capability dynamics |
| 3 | 6–7: oversight chunking, full queue |
| 4 | 8–9: industry, welfare ledger |
| 5 | 10: calibration and sensitivity |

Each later plan is written after the previous one lands.

## Global Constraints

- Work from the repo root `cel_skeleton/`. Run tests with `.venv/bin/python -m pytest tests -q`.
- Python 3.11. numpy ≥ 2.0, scipy ≥ 1.11, pytest ≥ 8 (verified with numpy 2.4.6, scipy 1.17.1, pytest 9.1.1).
- **Units (spec §0):**
  - Money is constant 2024 USD.
  - Labor quantities are per worker (`L_bar = 1` is one average worker).
  - One unit of task output is one worker-year at γ = 1.
  - `r` in the period economy is $ per delivered FLOP.
- Every parameter value lives in `calibration/parameters.csv`. No parameter literals in `cel/*.py`. Tests override values only through `S0.with_(...)`.
- Scenarios are diffs against `S0` (spec §0).
- Tests are pytest, gated by closed-form limiting cases (spec §11.3).
- The only randomness allowed is the Sobol grid, via `sobol_seed`.
- Catch specific exceptions (`Infeasible`), never bare `Exception`.
- Commit messages carry no AI attribution trailers (repo owner's standing instruction).

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `requirements.txt` | Pinned-floor dependencies | Create (Task 1) |
| `.gitignore` | Add `.venv/` | Modify (Task 1) |
| `calibration/parameters.csv` | Every parameter: value, range, prior, class, source, grade, notes | Create (Task 2), extend (Tasks 3, 5) |
| `cel/params.py` | `Params` fields and units; `load_params()`; `S0`; scenarios | Rewrite (Task 2), extend (Tasks 3, 5) |
| `cel/cohorts.py` | Displaced-worker queue with clearing function | Rewrite (Task 3) |
| `cel/grid.py` | Sobol grid with 4th dimension `a`; `capable_at(x)` | Rewrite (Task 4) |
| `cel/economy.py` | Period economy: capital nest, per-worker $ units, capability mask, brentq wage solve | Rewrite (Task 5) |
| `cel/capability.py` | `omega_train`, `omega_inf` decaying-rate paths | Create (Task 6) |
| `cel/compute_market.py` | `ComputeSupply` merit order, `solve_market`, regimes | Create (Task 7) |
| `cel/__init__.py` | Public exports | Modify (Tasks 2, 6, 7) |
| `cel/simulate.py` | Driver skeleton checklist | Modify (Task 7) |
| `README.md` | Layout and lessons | Rewrite (Task 7) |
| `tests/test_params.py` | CSV provenance and unit fixes | Create (Task 2) |
| `tests/test_queue.py` | Queue tests (moved from `test_steps_1_2.py`, extended) | Create (Task 3) |
| `tests/test_grid.py` | Grid tests (moved, extended) | Create (Task 4) |
| `tests/test_economy.py` | Step 1–2 tests (replaces `test_steps_1_2.py`) | Create (Task 5) |
| `tests/test_capability.py` | Ω path tests | Create (Task 6) |
| `tests/test_compute_market.py` | Merit order and regime tests | Create (Task 7) |

**Expected test counts after each task:**

| After task | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tests passing | 24 | 31 | 34 | 40 | 58 | 62 | 72 |

---

### Task 1: Baseline commit and environment

**Files:**
- Create: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a `.venv/` with dependencies, and a git baseline commit of the existing skeleton.

- [ ] **Step 1: Create `requirements.txt`**

**File: `requirements.txt` (full contents)**

```text
numpy>=2.0
scipy>=1.11
pytest>=8
```

- [ ] **Step 2: Ignore the virtualenv**

Append one line to `.gitignore`:

```text
.venv/
```

- [ ] **Step 3: Create the virtualenv and install**

Run: `python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt`
Expected: exits 0.

- [ ] **Step 4: Run the existing tests**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `24 passed`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore README.md cel tests
git commit -m "chore: import CEL skeleton (build steps 1-2) with requirements"
```

---

### Task 2: Parameter provenance CSV and unit fixes

Spec §10.6: `params.py` reads S0 from `calibration/parameters.csv`. Spec Appendix B, items 2–3, and §4.4 fix `m_H` (3 → 4, for the 75% margin), `eps0` (0.75e15 all-in), `kappa0` (4.0e-11, so a unit costs $30k), and `e0` (1e20 FLOP per worker-year). The existing tests still pass, because each one that depends on `e0` sets it explicitly.

**Files:**
- Create: `calibration/parameters.csv`
- Rewrite: `cel/params.py`
- Modify: `cel/__init__.py:2`
- Modify: `docs/superpowers/specs/2026-09-15-cel-model-design.md` (§10.1 class table)
- Test: `tests/test_params.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `cel.params.CSV_PATH: pathlib.Path`
  - `cel.params.load_params(path: Path = CSV_PATH) -> Params`, which raises `KeyError("parameters.csv out of sync: ...")` when fields and rows differ
  - `cel.params.S0 = load_params()`
  - `Params` fields now have no defaults. `int` fields are `t0_year, T, N_F, A_max, tau_c, M_tasks, sobol_seed`; all others are `float`.

- [ ] **Step 1: Write the failing test**

**File: `tests/test_params.py` (full contents)**

```python
"""Parameter provenance (spec §10.6) and unit fixes (spec Appendix B)."""
import csv
import dataclasses

import pytest

from cel.params import CSV_PATH, Params, S0, SCENARIOS, load_params, scenario

VALID_CLASSES = {"A", "B", "C", "D", "N"}
VALID_GRADES = {"A", "B", "C", "-"}


def _rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_csv_has_exactly_one_row_per_field():
    names = [r["name"] for r in _rows()]
    assert len(names) == len(set(names))
    assert set(names) == {f.name for f in dataclasses.fields(Params)}


def test_every_row_has_class_grade_and_source():
    for r in _rows():
        assert r["class"] in VALID_CLASSES, r["name"]
        assert r["grade"] in VALID_GRADES, r["name"]
        assert r["source"].strip(), r["name"]


def test_values_lie_inside_stated_ranges():
    for r in _rows():
        if r["lower"] and r["upper"]:
            lo, hi, v = float(r["lower"]), float(r["upper"]), float(r["value"])
            assert lo <= v <= hi, r["name"]


def test_s0_matches_csv_with_declared_types():
    rows = {r["name"]: r for r in _rows()}
    for f in dataclasses.fields(Params):
        v = getattr(S0, f.name)
        assert isinstance(v, int if f.type == "int" else float), f.name
        assert v == pytest.approx(float(rows[f.name]["value"])), f.name


def test_load_params_rejects_out_of_sync_csv(tmp_path):
    rows = _rows()[:-1]
    bad = tmp_path / "parameters.csv"
    with bad.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(KeyError, match="out of sync"):
        load_params(bad)


def test_scenarios_are_valid_diffs():
    for name in SCENARIOS:
        scenario(name)
    with pytest.raises(KeyError):
        S0.with_(not_a_param=1.0)


def test_unit_inconsistencies_are_fixed():
    assert 1.0 - 1.0 / S0.m_H == pytest.approx(0.75)                  # 75% gross margin
    assert S0.kappa0 * S0.eps0 * S0.unit_kW == pytest.approx(30_000.0)  # $ per 1 kW unit
    assert S0.e0 == pytest.approx(1e20)                                # worker-year anchor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_params.py -q`
Expected: collection error `ImportError: cannot import name 'CSV_PATH' from 'cel.params'`

- [ ] **Step 3: Create the CSV**

Class codes (spec §10.1, plus N):
- **A** measured
- **B** literature
- **C** structural unknown
- **D** identity or target
- **N** numerical setting

Grade codes (spec §10.3): **A** official/peer-reviewed, **B** research organization/industry, **C** press/assumption, **-** normalization or design.

Separators inside text fields are `;`, never `,`.

**File: `calibration/parameters.csv` (Task 2 contents)**

```text
name,value,lower,upper,prior,class,source,grade,notes
t0_year,2023,,,fixed,N,Model start year (design),-,
T,37,,,fixed,N,Horizon to 2060 (design),-,
eps0,0.75e15,0.5e15,1.4e15,loguniform,A,H100-class dense BF16 ~1e15 FLOP/s per GPU at ~1.3 kW all-in (NVIDIA datasheet; server specs),B,Provisional; verify all-in power (spec §10.2)
eps_headroom,30.0,10.0,300.0,loguniform,C,Physical efficiency ceiling relative to eps0 (assumption),C,Structural unknown
g_eps,0.28,0.20,0.40,uniform,B,Epoch AI ML hardware price-performance (doubling ~2.5 yr),B,
kappa0,4.0e-11,2.5e-11,6.0e-11,loguniform,A,~$30k per H100-class unit all-in (industry pricing reports),C,Provisional; kappa0*eps0*unit_kW = $30k
g_kappa0,0.30,0.20,0.40,uniform,D,Calibration check: must come out near g_eps + g_kappa_fab (spec §2.1),-,Not a free parameter
g_kappa_fab,0.03,0.0,0.06,uniform,B,Residual semiconductor fab learning (assumption),C,
m_H,4.0,2.0,5.0,uniform,A,NVIDIA data-center gross margin ~75% (10-K FY2025),A,m_H = 1/(1 - gross margin)
delta_phys,0.15,0.08,0.25,uniform,B,Accelerator physical retirement rate (assumption),C,
rho,0.10,0.06,0.14,uniform,B,Cost of capital for AI infrastructure (assumption),C,
util,0.5,0.3,0.8,uniform,B,Fleet average accelerator utilization (assumption),C,
unit_kW,1.0,,,fixed,D,Unit definition: one accelerator unit = 1 kW all-in (spec §2.5),-,
s_T,0.30,0.15,0.50,uniform,B,Training share of AI compute (assumption),C,
P_bar0_GW,15.0,5.0,20.0,uniform,A,AI-attributable power 2023 (check vs IEA Energy and AI 2025),C,Provisional; likely high (spec §4.2)
g_P0,0.45,0.25,0.60,uniform,C,Near-term AI power buildout rate (assumption),C,
g_P_inf,0.04,0.02,0.08,uniform,C,Long-run grid growth available to AI (assumption),C,
tau_P,6.0,3.0,12.0,uniform,C,Decay time of buildout rate (assumption),C,
PUE,1.2,1.1,1.5,uniform,A,Hyperscale PUE ~1.1-1.2 (LBNL 2024),B,
p_E,0.07,0.04,0.12,uniform,A,US industrial electricity price (EIA),A,
pi_E,0.01,0.0,0.03,uniform,B,Real drift in electricity price (assumption),C,
eta_E,0.03,0.0,0.08,uniform,C,Extra price drift when power binds (assumption),C,
iota0,0.38e-3,0.2e-3,0.5e-3,uniform,A,US grid average tCO2 per kWh (EPA eGRID),A,
g_iota,0.03,0.01,0.06,uniform,B,Grid decarbonization rate (assumption),C,
SCC,190.0,50.0,300.0,uniform,A,EPA 2023 social cost of carbon at 2% discount,A,
Theta,0.25,0.1,0.5,uniform,B,Frontier training run length ~3 months (Epoch AI),B,
g_Omega0,0.60,0.40,1.20,uniform,B,Algorithmic progress; Ho et al. 2024 suggest ~1.0/yr,B,S0 conservative (spec §10.2)
tau_Omega,8.0,4.0,20.0,uniform,C,Decay time of algorithmic progress (assumption),C,Structural unknown
g_Omega_inf,0.80,0.40,3.00,uniform,C,Inference efficiency; Epoch reports 9-900x/yr price declines at fixed capability,B,Structural unknown
x_t0,26.0,25.5,26.5,uniform,D,Calibration target: 2023 frontier run ~1e26 effective FLOP (Epoch AI),B,Pins varsigma (spec §4.2)
x50,29.0,27.0,33.0,uniform,C,log10 compute for 50% task coverage; calibrate to METR time horizons,C,Structural unknown
s_diff,1.5,0.5,3.0,uniform,C,Difficulty spread in decades per logit; calibrate to METR time horizons,C,Structural unknown
phi_max,0.75,0.40,0.98,uniform,C,Automatable ceiling (assumption),C,Structural unknown; 1 - phi_max floors ell
e0,1.0e20,1.0e19,1.0e21,loguniform,C,Inference FLOP per worker-year at median difficulty: ~1e8 tokens x ~1e12 FLOP/token (spec §4.4),C,Structural unknown
theta,1.0,0.3,1.5,uniform,C,Inference-cost elasticity to difficulty per decade (assumption),C,Drives Proposition 2b
psi,0.5,0.0,3.0,uniform,C,Clayton dependence of step errors (assumption),C,Used from build step 6
lam,0.10,0.0,1.0,uniform,C,Error propagation per downstream chunk (assumption),C,Used from build step 6
rho_conceal,0.15,0.0,0.4,uniform,C,Self-concealing error share (assumption),C,Used from build step 6
q_bar,0.90,0.6,0.99,uniform,C,Detection probability of non-concealed errors (assumption),C,Used from build step 6
omega0,0.30,0.1,0.5,uniform,D,Calibration target: oversight labor share at t0 (assumption),C,Target not input
sigma,0.60,0.30,1.50,uniform,B,Task elasticity of substitution (Acemoglu-Restrepo 2018; Oberfield-Raval 2021),B,Threshold at 1
nu,0.005,0.0,0.02,uniform,C,New-task creation rate (Autor et al. 2024),B,
A_Y,1.0,,,fixed,D,TFP scale (normalization until the Task 5 calibration identity),-,
N0,1.0,,,fixed,N,Initial task measure (normalization),-,
L_bar,1.0,,,fixed,D,Labor per worker (normalization: one average worker),-,
eps_L,0.30,0.1,0.6,uniform,B,Labor supply elasticity (Chetty et al. 2011),A,
w_res_ratio,0.50,0.3,0.8,uniform,C,Wage floor relative to initial wage (assumption),C,
N_F,5,2,10,uniform,A,Number of frontier model developers (industry count),B,
F0_trillion,0.6,0.3,1.0,uniform,C,Financing cap at 2026 in $T/yr (hyperscaler capex guidance),C,Press-grade
A_max,20,,,fixed,N,Queue-age cohorts tracked (design),-,
mu0,0.02,0.005,0.05,uniform,C,Initial retraining capacity share of labor per yr (assumption),C,
tau_c,4,1,8,uniform,C,Capacity build lag in years (assumption),C,
chi,2.0,1.0,5.0,uniform,C,Capacity adjustment curvature (assumption),C,
xi_reemp,0.35,0.1,0.6,uniform,B,Negative duration dependence (Kroft-Lange-Notowidigdo 2013),A,
pi_scar0,0.03,0.0,0.08,uniform,C,Base permanent-exit hazard (assumption),C,
pi_scar_slope,0.02,0.0,0.05,uniform,C,Exit hazard increase per year queued (assumption),C,
turnover_ceiling,0.025,0.015,0.04,uniform,B,Displacement absorbed by retirements and entry per yr (assumption),C,
c2_burst,1.0,0.5,4.0,uniform,C,Kingman burstiness (assumption),C,
beta,0.97,0.95,0.99,uniform,B,Social discount factor,B,
eta_ia,1.5,0.5,3.0,uniform,B,Inequality aversion,B,
g0,0.025,0.015,0.035,uniform,B,No-AI world growth (IMF WEO long-run),B,
Y0_trillion,105.0,100.0,110.0,uniform,A,World GDP 2023 in $T (World Bank WDI),A,
M_tasks,4096,,,fixed,N,Sobol task grid size (power of two),-,
sobol_seed,7,,,fixed,N,Sobol scramble seed,-,
kappaE_mu,0.0,-1.0,1.0,uniform,C,Log error consequence relative to task value (assumption),C,Used from build step 6
kappaE_sig,1.0,0.5,2.0,uniform,C,Dispersion of error consequence (assumption),C,Used from build step 6
Gamma_mu,0.0,-1.0,1.0,uniform,C,Log modularity cost (assumption),C,Used from build step 6
Gamma_sig,0.8,0.3,1.5,uniform,C,Dispersion of modularity cost (assumption),C,Used from build step 6
```

- [ ] **Step 4: Rewrite `cel/params.py`**

**File: `cel/params.py` (Task 2 contents)**

```python
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
```

`f.type` is the string `"int"` or `"float"` because of `from __future__ import annotations`. That is why `load_params` compares against a string.

- [ ] **Step 5: Export `load_params`**

In `cel/__init__.py`, replace
`from .params import Params, S0, SCENARIOS, scenario`
with
`from .params import Params, S0, SCENARIOS, scenario, load_params`

- [ ] **Step 6: Record class N in the spec**

In `docs/superpowers/specs/2026-09-15-cel-model-design.md` §10.1, replace the row

```text
| D. Identities | A_Y, ς, `omega0` target, `g_kappa0` check | Solved, not chosen |
```

with

```text
| D. Identities | A_Y, ς, `omega0` target, `g_kappa0` check | Solved, not chosen |
| N. Numerical settings | t0_year, T, M_tasks, sobol_seed, A_max, N0 | Fixed design choices, not calibrated |
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `31 passed`

- [ ] **Step 8: Commit**

```bash
git add calibration/parameters.csv cel/params.py cel/__init__.py tests/test_params.py docs/superpowers/specs/2026-09-15-cel-model-design.md
git commit -m "feat: load parameters from provenance CSV and fix unit inconsistencies"
```

---

### Task 3: Queue clearing function (fixes the saturation discontinuity)

Spec §7.3 and Appendix B item 1. The current congestion factor depends on the inflow ratio u and is switched off at u ≥ 1. The replacement is a clearing function of the *employable* stock:

```text
X = μ·S_eff/(S_eff + c2·μ/h0),     S_eff = Σ_a S_a·e^{−ξa}
```

X is allocated across ages in proportion to S_a·e^{−ξa}.
- With ξ = 0 the steady state is W = c2/(h0(1−u)). At c2 = 1 that is exactly the Kingman values the old tests used.
- With ξ > 0, an aging pool lowers S_eff and amplifies congestion.

Near saturation the queue converges slowly (rate 1 − dX/dS), so Kingman checks need 5000 steps; at u = 0.95, 400 steps reaches only 16.9 against 20.

**Files:**
- Modify: `calibration/parameters.csv` (add `h0_reemp` row)
- Modify: `cel/params.py` (add `h0_reemp` field)
- Rewrite: `cel/cohorts.py`
- Modify: `tests/test_steps_1_2.py` (remove queue tests; they move)
- Modify: `docs/superpowers/specs/2026-09-15-cel-model-design.md` (§7.3, Appendix A.2)
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `S0.with_`, `Params.h0_reemp: float`.
- Produces: `DisplacedPool(p).step(inflow: float, capacity: float, match_cap: float | None = None) -> QueueStep`. The signature is unchanged. `QueueStep` fields are unchanged: `outflow, exits, stock, mean_duration, saturation`.

- [ ] **Step 1: Write the failing test**

**File: `tests/test_queue.py` (full contents)**

```python
"""Displaced-worker queue with the clearing function (spec §7.3)."""
import numpy as np
import pytest

from cel import S0, DisplacedPool


def _pure_queue():
    # no scarring exits, no duration dependence: isolates congestion
    return S0.with_(pi_scar0=0.0, pi_scar_slope=0.0, xi_reemp=0.0, c2_burst=1.0, h0_reemp=1.0)


def _run(p, inflow, capacity, steps):
    q = DisplacedPool(p)
    for _ in range(steps):
        st = q.step(inflow=inflow, capacity=capacity)
    return st


def test_queue_matches_kingman_at_unit_burstiness():
    # steady state W = c2 / (h0 (1 - u)) = 1 + u/(1-u) at c2 = h0 = 1.
    # Near saturation convergence is slow (rate 1 - dX/dS), hence 5000 steps.
    for inflow, expected in [(0.005, 4 / 3), (0.010, 2.0), (0.015, 4.0), (0.019, 20.0)]:
        st = _run(_pure_queue(), inflow, 0.02, 5000)
        assert st.mean_duration == pytest.approx(expected, rel=0.02), (inflow, st.mean_duration)


def test_queue_duration_scales_with_burstiness():
    base = _run(_pure_queue(), 0.010, 0.02, 5000).mean_duration
    bursty = _run(_pure_queue().with_(c2_burst=2.0), 0.010, 0.02, 5000).mean_duration
    assert bursty == pytest.approx(2.0 * base, rel=0.02)


def test_queue_duration_is_continuous_and_monotone_through_saturation():
    inflows = [0.0199, 0.01999, 0.02, 0.02001, 0.0201]
    durations = [_run(_pure_queue(), lam, 0.02, 200).mean_duration for lam in inflows]
    assert all(np.diff(durations) > 0), durations
    # regression: the inflow-ratio factor returned 1.0 yr at u = 1
    assert durations[2] > 10.0


def test_queue_saturated_grows_without_bound():
    q = DisplacedPool(_pure_queue())
    stocks = [q.step(inflow=0.03, capacity=0.02).stock for _ in range(40)]
    assert stocks[-1] > stocks[-10] > stocks[-20]
    assert q.step(0.03, 0.02).mean_duration > 10.0


def test_queue_duration_dependence_amplifies_congestion():
    # with xi > 0 the pool ages into low-employability cohorts, S_eff falls
    # and duration exceeds the Kingman value (2.0 at u = 0.5)
    dep = S0.with_(pi_scar0=0.0, pi_scar_slope=0.0, xi_reemp=0.35, c2_burst=1.0, h0_reemp=1.0)
    st = _run(dep, 0.010, 0.02, 200)
    assert st.mean_duration > 2.0


def test_queue_outflow_never_exceeds_stock_or_capacity():
    q = DisplacedPool(_pure_queue().with_(c2_burst=0.5))   # c2 < h0: per-head rate can exceed 1
    for _ in range(50):
        before = q.S.sum()
        st = q.step(inflow=0.001, capacity=0.02)
        assert st.outflow <= 0.02 + 1e-15
        assert st.outflow <= before + 1e-15
```

- [ ] **Step 2: Remove the old queue tests from `tests/test_steps_1_2.py`**

- Delete everything from the line `# ---------------------------------------------------------------- cohort queue` to the end of the file. That removes `_pure_queue`, `test_queue_kingman_exact_without_duration_dependence`, `test_queue_saturated_grows_without_bound` and `test_queue_duration_dependence_amplifies_congestion`.
- In the import line, replace `, Infeasible, DisplacedPool` with `, Infeasible`.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_queue.py -q`
Expected: FAIL with `KeyError: "unknown parameter(s): ['h0_reemp']"`

- [ ] **Step 4: Add the `h0_reemp` parameter**

In `calibration/parameters.csv`, insert this row directly after the `xi_reemp,...` row:

```text
h0_reemp,1.0,0.5,3.0,uniform,C,Base re-employment hazard per yr; normalization giving W = c2/(1-u) (spec §7.3),C,
```

In `cel/params.py`, insert this line directly after the `xi_reemp: float ...` line:

```python
    h0_reemp: float                  # base re-employment hazard /yr
```

- [ ] **Step 5: Rewrite `cel/cohorts.py`**

**File: `cel/cohorts.py` (full contents)**

```python
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
```

- [ ] **Step 6: Amend the spec**

In §7.3, replace the formula line

```text
X(S) = μ_eff · S / (S + c2·μ_eff/h0)
```

with

```text
X = μ_eff · S_eff / (S_eff + c2·μ_eff/h0),     S_eff = Σ_a S_a·e^{−ξ·a}
```

and replace the sentence

```text
Outflow X(S) is allocated across ages in proportion to S_a·e^{−ξ·a} (negative duration dependence).
```

with

```text
Outflow X is allocated across ages in proportion to S_a·e^{−ξ·a} (negative duration dependence). Because X depends on the employable stock S_eff rather than the raw stock, an aging pool clears more slowly; with ξ = 0, S_eff = S and the steady-state formula below holds exactly.
```

In Appendix A.2, insert this row directly after the `g_T` row:

```text
| h0_reemp | Base re-employment hazard (/yr); W = c2/(h0(1 − u)) | 1.0 | [0.5, 3.0] | C |
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `34 passed`

- [ ] **Step 8: Commit**

```bash
git add calibration/parameters.csv cel/params.py cel/cohorts.py tests/test_queue.py tests/test_steps_1_2.py docs/superpowers/specs/2026-09-15-cel-model-design.md
git commit -m "fix: replace queue congestion factor with a stock-based clearing function"
```

---

### Task 4: Automatability flag independent of difficulty

Spec §4.3 and Appendix B item 6. A fourth Sobol dimension gives each task a flag `a ~ Bernoulli(phi_max)`, independent of difficulty `d`. `capable_at(x) = (d <= x) & a`. `rank` stays as a diagnostic; the economy stops using it in Task 5. Sobol one-dimensional projections are stratified, so the share of flagged tasks is within 1/M of `phi_max`.

**Files:**
- Rewrite: `cel/grid.py`
- Modify: `tests/test_steps_1_2.py` (remove grid tests; they move)
- Test: `tests/test_grid.py`

**Interfaces:**
- Consumes: `Params.phi_max`, `Params.sobol_seed`.
- Produces:
  - `TaskGrid.a: np.ndarray[bool]` of shape `(M,)`
  - `TaskGrid.capable_at(x: float) -> np.ndarray[bool]`
  - `TaskGrid.rank`, `coverage_at` and `inference_flop` are unchanged.

- [ ] **Step 1: Write the failing test**

**File: `tests/test_grid.py` (full contents)**

```python
"""Task grid: logistic coverage and the automatability flag (spec §4.3)."""
import numpy as np
import pytest

from cel import S0, TaskGrid


def test_grid_reproduces_logistic_H():
    p = S0
    g = TaskGrid(p)
    for x in [p.x50 - 2 * p.s_diff, p.x50, p.x50 + 2 * p.s_diff]:
        H = 1.0 / (1.0 + np.exp(-(x - p.x50) / p.s_diff))
        assert abs(g.coverage_at(x) - H) < 0.01, (x, g.coverage_at(x), H)


def test_grid_rank_is_uniform_and_ordered_by_d():
    g = TaskGrid(S0)
    order = np.argsort(g.d)
    assert np.all(np.diff(g.rank[order]) > 0)
    assert abs(g.rank.mean() - 0.5) < 1e-6


@pytest.mark.parametrize("phi_max", [0.4, 0.75, 0.98])
def test_automatable_share_equals_phi_max(phi_max):
    g = TaskGrid(S0.with_(phi_max=phi_max))
    # Sobol 1-D projections are stratified: one point per 1/M interval
    assert abs(g.a.mean() - phi_max) <= 1.0 / g.M + 1e-12


def test_automatable_flag_is_independent_of_difficulty():
    g = TaskGrid(S0)
    easy = g.rank < 0.5
    assert abs(g.a[easy].mean() - g.a[~easy].mean()) < 0.03


def test_capable_at_combines_threshold_and_flag():
    g = TaskGrid(S0)
    cap = g.capable_at(S0.x50)
    assert np.array_equal(cap, (g.d <= S0.x50) & g.a)
    assert cap.mean() == pytest.approx(S0.phi_max * 0.5, abs=0.02)


def test_phi_max_is_a_hard_ceiling_on_capability():
    g = TaskGrid(S0)
    assert g.capable_at(1e9).mean() == pytest.approx(g.a.mean())
```

- [ ] **Step 2: Remove the old grid tests from `tests/test_steps_1_2.py`**

Delete the block from the line `# ---------------------------------------------------------------- grid sanity` up to, but not including, the line `# ---------------------------------------------------------------- step 1`. That removes `test_grid_reproduces_logistic_H` and `test_grid_rank_is_uniform_and_ordered_by_d`.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_grid.py -q`
Expected: FAIL with `AttributeError: 'TaskGrid' object has no attribute 'a'`

- [ ] **Step 4: Rewrite `cel/grid.py`**

**File: `cel/grid.py` (full contents)**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `40 passed`

- [ ] **Step 6: Commit**

```bash
git add cel/grid.py tests/test_grid.py tests/test_steps_1_2.py
git commit -m "feat: add automatability flag independent of task difficulty"
```

---

### Task 5: Period economy with capital nest, per-worker dollar units, and capability mask

Spec §5.1–5.3 (build steps 1–2 redone), plus Appendix B items 4 and 6.
- **Output:** `Y = A_Y·K_o^α·T^(1−α)`.
- **Labor clearing** gives `T = L/ell_T(w)`.
- **The wage** solves `log(P_T/MPT) = 0` with brentq in log w. This residual increases in w, and with α = 0 it reduces exactly to the old `P_T = A_Y` condition.
- **The capability mask** replaces `rank < phi_cap`.

**Calibration identity.** Because the outer nest is Cobb–Douglas, the no-AI labor share is 1 − α_K. So `s_L0 = 0.55` pins **α_K = 0.45**; the spec's provisional 0.40 is amended here. `A_Y = Y0/L_workers = $30,000` per worker-year, which gives a no-AI wage of $16,500.

**Files:**
- Modify: `calibration/parameters.csv` (add `alpha_K` and `L_workers`; change the `A_Y` row)
- Modify: `cel/params.py` (add `alpha_K` and `L_workers` fields)
- Rewrite: `cel/economy.py`
- Delete: `tests/test_steps_1_2.py`
- Modify: `docs/superpowers/specs/2026-09-15-cel-model-design.md` (§5.1, Appendix A.2)
- Test: `tests/test_economy.py`

**Interfaces:**
- Consumes:
  - `TaskGrid.capable_at`, `TaskGrid.rank`, `TaskGrid.inference_flop`
  - `Params.alpha_K`, `Params.A_Y`, `Params.L_workers`
- Produces:
  - `solve_period(grid: TaskGrid, p: Params, r: float, capable: np.ndarray, K_o: float = 1.0, L_eff: float | None = None, Omega_inf: float = 1.0, force: bool = False) -> PeriodResult`. `r` is $ per delivered FLOP.
  - `PeriodResult` fields: `w, r, Y, T_agg, P_T, L_d, K_d, s_L, phi_eff, prices, auto, nominal_check, mpt_check`. `K_d` is delivered FLOP per worker-year. The old `phi_cap` and `P_Y` fields are removed.
  - `task_prices(grid, w, c_AI, capable, force=False) -> (prices, auto)`
  - `task_price_index(grid, p, prices) -> float`, which raises `ValueError` at σ = 1
  - `labor_share_closed_form(p, phi, w, p_A, N=1.0) -> float`, now including the factor (1 − α_K)
  - `Infeasible(RuntimeError)`, raised only when α_K = 0

- [ ] **Step 1: Write the failing test, replacing the old step 1–2 file**

Run: `git rm -q tests/test_steps_1_2.py`

**File: `tests/test_economy.py` (full contents)**

```python
"""
Limiting-case tests that gate build steps 1 and 2 (spec §5.1-5.3).

Step 1: no AI. With alpha_K = 0 and N = L_bar = 1:  w = Y = A_Y, s_L = 1.
        With alpha_K > 0 and capital K_o:  Y = A_Y K_o^alpha, w = (1-alpha) Y,
        s_L = 1 - alpha.
Step 2: exogenous capability at fixed r, symmetric tasks (theta = 0 so every
        automated task has the same AI price p_A). Proposition 3 must hold
        exactly, scaled by (1 - alpha_K), with the right limits in sigma.
"""
import numpy as np
import pytest

from cel import S0, TaskGrid, solve_period, labor_share_closed_form, Infeasible


def _nothing_capable(g):
    return np.zeros(g.M, dtype=bool)


# ---------------------------------------------------------------- step 1
@pytest.mark.parametrize("A_Y", [1.0, 2.5])
@pytest.mark.parametrize("sigma", [0.6, 1.5])
def test_step1_no_ai_closed_form_without_capital(A_Y, sigma):
    p = S0.with_(A_Y=A_Y, sigma=sigma, alpha_K=0.0)
    g = TaskGrid(p)
    res = solve_period(g, p, r=1.0, capable=_nothing_capable(g))
    assert res.phi_eff == 0.0 and res.K_d == 0.0
    assert res.w == pytest.approx(A_Y, rel=1e-8)
    assert res.Y == pytest.approx(A_Y, rel=1e-8)
    assert res.L_d == pytest.approx(p.L_bar, rel=1e-8)
    assert res.s_L == pytest.approx(1.0, rel=1e-8)
    assert abs(res.mpt_check) < 1e-8
    assert abs(res.nominal_check) < 1e-8


@pytest.mark.parametrize("alpha_K", [0.3, 0.45])
@pytest.mark.parametrize("sigma", [0.6, 1.5])
def test_step1_no_ai_closed_form_with_capital(alpha_K, sigma):
    p = S0.with_(sigma=sigma, alpha_K=alpha_K)
    g = TaskGrid(p)
    K_o = 2.0
    res = solve_period(g, p, r=1.0, capable=_nothing_capable(g), K_o=K_o)
    Y = p.A_Y * K_o ** alpha_K
    assert res.Y == pytest.approx(Y, rel=1e-8)
    assert res.w == pytest.approx((1.0 - alpha_K) * Y, rel=1e-8)
    assert res.s_L == pytest.approx(1.0 - alpha_K, rel=1e-8)
    assert res.L_d == pytest.approx(p.L_bar, rel=1e-8)
    assert abs(res.mpt_check) < 1e-8


def test_s0_calibration_identity():
    # spec §5.1: Y0 = $105T over 3.5e9 workers, labor share 0.55
    assert S0.A_Y == pytest.approx(S0.Y0_trillion * 1e12 / S0.L_workers)
    g = TaskGrid(S0)
    res = solve_period(g, S0, r=1.0, capable=_nothing_capable(g))
    assert res.Y == pytest.approx(30_000.0, rel=1e-9)
    assert res.w == pytest.approx(16_500.0, rel=1e-9)
    assert res.s_L == pytest.approx(0.55, rel=1e-9)


# ---------------------------------------------------------------- step 2
def _symmetric(sigma, phi, p_A, alpha_K=0.0):
    # theta = 0 -> e(i) = e0 for all i; e0 = 1 so c_AI = r = p_A.
    # A_Y = 1/(1-alpha_K) keeps the no-AI wage at 1 for every alpha_K.
    p = S0.with_(sigma=sigma, theta=0.0, e0=1.0, alpha_K=alpha_K, A_Y=1.0 / (1.0 - alpha_K))
    g = TaskGrid(p)
    res = solve_period(g, p, r=p_A, capable=g.rank < phi, force=True)
    return p, res


@pytest.mark.parametrize("alpha_K", [0.0, 0.45])
@pytest.mark.parametrize("sigma", [0.4, 0.6, 0.9, 1.5])
@pytest.mark.parametrize("phi", [0.1, 0.5, 0.9])
def test_step2_proposition3_exact(alpha_K, sigma, phi):
    # alpha_K = 0: p_A = 0.9 stays inside the feasibility bound for all cases.
    # alpha_K > 0: forcing 90% of tasks onto AI priced above the wage pushes
    # the wage below p_A, so use an AI price that is actually cheaper.
    p_A = 0.9 if alpha_K == 0.0 else 0.1
    p, res = _symmetric(sigma, phi, p_A, alpha_K)
    assert res.phi_eff == pytest.approx(phi, abs=1.0 / p.M_tasks)
    assert np.all(res.prices[res.auto] == p_A)
    assert p_A < res.w              # cost condition holds even though forced
    expected = labor_share_closed_form(p, res.phi_eff, res.w, p_A)
    assert res.s_L == pytest.approx(expected, rel=1e-8)
    assert abs(res.nominal_check) < 1e-8
    assert abs(res.mpt_check) < 1e-8
    assert res.L_d == pytest.approx(p.L_bar, rel=1e-8)


def test_step2_sigma_below_one_labor_share_to_one():
    phi = 0.5
    shares = [_symmetric(0.6, phi, p_A)[1].s_L for p_A in [0.5, 0.1, 0.01, 1e-4]]
    assert all(np.diff(shares) > 0)                # rising as AI cheapens
    assert shares[-1] > 0.95                       # Baumol: -> 1


def test_step2_sigma_above_one_labor_share_falls():
    phi = 0.5
    shares = [_symmetric(1.5, phi, p_A)[1].s_L for p_A in [1.0, 0.6, 0.35]]
    assert all(np.diff(shares) < 0)                # falling as AI cheapens
    # alpha_K = 0: below phi * p_A^(1-sigma) = A_Y^(1-sigma) the numeraire fails
    with pytest.raises(Infeasible):
        _symmetric(1.5, phi, 0.2)


def test_step2_capital_share_removes_infeasibility():
    # same point with alpha_K > 0: T -> inf drives the marginal product down
    p, res = _symmetric(1.5, 0.5, 0.2, alpha_K=0.45)
    assert np.isfinite(res.w)
    assert abs(res.mpt_check) < 1e-8


def test_step2_deployment_wedge_when_not_forced():
    # theta > 0: hard tasks are expensive to run, so phi_eff < capable share
    p = S0.with_(sigma=0.6, theta=1.0, e0=1.0, alpha_K=0.0, A_Y=1.0)
    g = TaskGrid(p)
    res = solve_period(g, p, r=0.3, capable=g.rank < 0.9)
    assert 0.0 < res.phi_eff < 0.9
    assert np.all(res.prices[res.auto] < res.w / g.gamma[res.auto])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_economy.py -q`
Expected: FAIL with `KeyError: "unknown parameter(s): ['alpha_K']"` (or `TypeError` on `capable=`)

- [ ] **Step 3: Add the parameters**

In `calibration/parameters.csv`:
- Insert after the `nu,...` row:
  ```text
  alpha_K,0.45,0.35,0.50,uniform,A,1 - global labor share ~0.55 (Penn World Table 10),B,Sets s_L0 = 1 - alpha_K
  ```
- Replace the `A_Y,...` row with:
  ```text
  A_Y,30000.0,,,fixed,D,Identity: Y0_trillion * 1e12 / L_workers at K_o = 1 (spec §5.1),-,
  ```
- Insert after the `L_bar,...` row:
  ```text
  L_workers,3.5e9,3.3e9,3.7e9,uniform,A,Global labour force (ILO),A,
  ```

In `cel/params.py`:
- Insert after the `nu: float ...` line:
  ```python
      alpha_K: float                   # conventional capital share (outer Cobb-Douglas)
  ```
- Insert after the `L_bar: float ...` line:
  ```python
      L_workers: float                 # global labor force (workers)
  ```

- [ ] **Step 4: Rewrite `cel/economy.py`**

**File: `cel/economy.py` (full contents)**

```python
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
```

- [ ] **Step 5: Amend the spec**

In §5.1, replace

```text
K_o and A_Y grow at the no-AI rate g0. A_Y is calibrated so that Y(t0) = $105T and s_L(t0) = 0.55, implying a global average wage of ≈$16k per worker-year.
```

with

```text
K_o and A_Y grow at the no-AI rate g0. Output is per worker: with K_o(t0) = 1, A_Y = Y0/L_workers = $30,000 per worker-year. Because the outer nest is Cobb–Douglas, the no-AI labor share is 1 − α_K, so s_L(t0) = 0.55 pins α_K = 0.45 and the average wage at ≈$16,500 per worker-year.
```

In Appendix A.2, replace the row

```text
| alpha_K | Conventional capital share | 0.40 | [0.30, 0.45] | A |
```

with

```text
| alpha_K | Conventional capital share (= 1 − s_L0) | 0.45 | [0.35, 0.50] | A |
```

and insert this row directly after the `h0_reemp` row:

```text
| L_workers | Global labor force (workers); converts per-worker quantities to totals | 3.5e9 | [3.3e9, 3.7e9] | A |
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `58 passed`

- [ ] **Step 7: Commit**

```bash
git add calibration/parameters.csv cel/params.py cel/economy.py tests/test_economy.py docs/superpowers/specs/2026-09-15-cel-model-design.md
git commit -m "feat: capital nest, per-worker dollar units and capability mask in period economy"
```

---

### Task 6: Algorithmic-progress paths

Spec §4.1 and Appendix B item 5. Both multipliers grow at decaying rates that share the timescale τ_Ω, so inference efficiency is bounded (about 600× at S0) instead of about 10¹³× by 2060.

**Files:**
- Create: `cel/capability.py`
- Modify: `cel/__init__.py` (append an export)
- Test: `tests/test_capability.py`

**Interfaces:**
- Consumes: `Params.g_Omega0`, `Params.g_Omega_inf`, `Params.tau_Omega`.
- Produces: `omega_train(p: Params, tau: float) -> float` and `omega_inf(p: Params, tau: float) -> float`, where `tau` is years since t0. Plan 2 feeds `omega_inf` into `solve_period(..., Omega_inf=...)`.

- [ ] **Step 1: Write the failing test**

**File: `tests/test_capability.py` (full contents)**

```python
"""Algorithmic progress paths (spec §4.1)."""
import math

import pytest

from cel import S0, omega_train, omega_inf


def test_multipliers_start_at_one():
    assert omega_train(S0, 0.0) == 1.0
    assert omega_inf(S0, 0.0) == 1.0


def test_initial_growth_rate_equals_g():
    h = 1e-6
    assert math.log(omega_train(S0, h)) / h == pytest.approx(S0.g_Omega0, rel=1e-4)
    assert math.log(omega_inf(S0, h)) / h == pytest.approx(S0.g_Omega_inf, rel=1e-4)


def test_cumulative_limits_match_spec():
    # spec §4.1: at S0, ~120x training and ~600x inference in total
    assert omega_train(S0, 1e4) == pytest.approx(math.exp(S0.g_Omega0 * S0.tau_Omega), rel=1e-9)
    assert omega_inf(S0, 1e4) == pytest.approx(math.exp(S0.g_Omega_inf * S0.tau_Omega), rel=1e-9)
    assert 115 < omega_train(S0, 1e4) < 125
    assert 590 < omega_inf(S0, 1e4) < 610


def test_inference_efficiency_is_bounded_by_2060():
    # a constant 0.80/yr would give exp(0.8 * 37) ~ 7e12 by 2060
    assert omega_inf(S0, 37.0) < 1e3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_capability.py -q`
Expected: collection error `ImportError: cannot import name 'omega_train' from 'cel'`

- [ ] **Step 3: Create `cel/capability.py`**

**File: `cel/capability.py` (full contents)**

```python
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
```

- [ ] **Step 4: Export**

Append to `cel/__init__.py`:

```python
from .capability import omega_train, omega_inf
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `62 passed`

- [ ] **Step 6: Commit**

```bash
git add cel/capability.py cel/__init__.py tests/test_capability.py
git commit -m "feat: algorithmic-progress paths with decaying growth rates"
```

---

### Task 7: Merit-order compute market (build step 3), docs, and checklist

Spec §9.2, build step 3.
- **Supply** is a merit order of vintages, each with (capacity, operating cost).
- **Clearing price:** the smallest r with `K_d(r) ≤ S(r)`, found by bisection in log r. Both curves are step functions, so Brent is the wrong tool for the outer loop; the inner wage solve stays brentq. This swaps the nesting order written in spec §9.2, and Step 6 amends the spec.
- **Regime:** measured against the *marginal dispatched* vintage, so a price between two operating costs is correctly labeled scarce, with the dearer vintage idle.
- **Infeasible:** when the inner solve raises `Infeasible`, the market treats it as unbounded demand. That is the step-3 gate: "Infeasible never raised".

**Files:**
- Create: `cel/compute_market.py`
- Rewrite: `cel/__init__.py`
- Rewrite: `cel/simulate.py`
- Rewrite: `README.md`
- Modify: `docs/superpowers/specs/2026-09-15-cel-model-design.md` (§9.2)
- Test: `tests/test_compute_market.py`

**Interfaces:**
- Consumes: `solve_period`, `PeriodResult`, `Infeasible` (Task 5); `TaskGrid.capable_at` (Task 4).
- Produces:
  - `ComputeSupply(capacity: np.ndarray, op_cost: np.ndarray)`, with properties `.total -> float` and `.at(r) -> float`. It raises `ValueError` on mismatched shapes, negative capacity, or non-positive cost. Capacity is delivered FLOP per worker-year; cost is $ per delivered FLOP.
  - `solve_market(grid, p, supply, capable, K_o=1.0, L_eff=None, Omega_inf=1.0, rtol=1e-12, max_expand=200) -> MarketResult`
  - `MarketResult(period: PeriodResult, r: float, regime: str, utilization: float, scarcity_rent: float)`, where `regime ∈ {"slack", "scarce"}`
  - `MarketError(RuntimeError)`, raised if no price up to 2^200 × min cost clears
  - Plan 2 builds `ComputeSupply` from vintages: capacity = C_inf·Y_s/L_workers per vintage, op_cost = energy cost per delivered FLOP.

- [ ] **Step 1: Write the failing test**

**File: `tests/test_compute_market.py` (full contents)**

```python
"""Compute market with merit-order supply (spec §9.2, build step 3)."""
import numpy as np
import pytest

from cel import S0, TaskGrid, ComputeSupply, Infeasible, solve_market, solve_period

OP = 1e-18    # $ per delivered FLOP: roughly an H100-hour at $2 over its delivered FLOP


@pytest.fixture(scope="module")
def s0_economy():
    g = TaskGrid(S0)
    capable = g.capable_at(30.0)            # about half of tasks capable
    D0 = solve_period(g, S0, OP, capable).K_d
    return g, capable, D0


def test_supply_is_a_merit_order():
    s = ComputeSupply(capacity=np.array([1.0, 2.0, 3.0]), op_cost=np.array([3.0, 1.0, 2.0]))
    assert s.total == 6.0
    assert s.at(0.5) == 0.0
    assert s.at(1.0) == 2.0
    assert s.at(2.5) == 5.0
    assert s.at(10.0) == 6.0


def test_supply_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ComputeSupply(capacity=np.array([1.0, 2.0]), op_cost=np.array([1.0]))
    with pytest.raises(ValueError):
        ComputeSupply(capacity=np.array([1.0]), op_cost=np.array([0.0]))


def test_compute_demand_is_nonincreasing_in_price(s0_economy):
    g, capable, _ = s0_economy
    kd = [solve_period(g, S0, r, capable).K_d for r in OP * np.logspace(-2, 4, 25)]
    assert np.all(np.diff(kd) <= 1e-12 * max(kd))


@pytest.mark.parametrize("factor, regime", [(0.5, "scarce"), (0.9, "scarce"),
                                            (1.1, "slack"), (2.0, "slack")])
def test_regime_switches_where_demand_meets_capacity(s0_economy, factor, regime):
    g, capable, D0 = s0_economy
    m = solve_market(g, S0, ComputeSupply(np.array([factor * D0]), np.array([OP])), capable)
    assert m.regime == regime
    if regime == "slack":
        assert m.r == OP
        assert m.scarcity_rent == 0.0
        assert m.utilization == pytest.approx(1.0 / factor, rel=1e-9)
    else:
        assert m.r > OP
        assert m.scarcity_rent == pytest.approx(m.r - OP)
        assert m.period.K_d <= factor * D0 * (1.0 + 1e-9)
        assert m.utilization > 0.999
        # any lower price would overload capacity
        assert solve_period(g, S0, m.r * (1.0 - 1e-6), capable).K_d > factor * D0


def test_price_set_by_marginal_vintage(s0_economy):
    # cheap vintage too small, dear vintage large: price = dear vintage's cost
    g, capable, D0 = s0_economy
    supply = ComputeSupply(np.array([0.3 * D0, 10.0 * D0]), np.array([OP, 3.0 * OP]))
    m = solve_market(g, S0, supply, capable)
    assert m.regime == "slack"
    assert m.r == pytest.approx(3.0 * OP, rel=1e-9)
    assert m.scarcity_rent == 0.0


def test_scarcity_between_vintages_leaves_dear_vintage_idle(s0_economy):
    # demand at 3*OP is below 0.5*D0, so the price settles between the two
    # operating costs: the cheap vintage earns a scarcity rent, the dear one idles
    g, capable, D0 = s0_economy
    supply = ComputeSupply(np.array([0.5 * D0, 10.0 * D0]), np.array([OP, 3.0 * OP]))
    m = solve_market(g, S0, supply, capable)
    assert m.regime == "scarce"
    assert OP < m.r < 3.0 * OP
    assert m.scarcity_rent == pytest.approx(m.r - OP)
    assert m.period.K_d <= 0.5 * D0 * (1.0 + 1e-9)


def test_infeasible_never_escapes_market_clearing():
    # alpha_K = 0 and sigma > 1: at the operating cost the numeraire fails
    p = S0.with_(sigma=1.5, alpha_K=0.0, A_Y=1.0, theta=1.0, e0=1.0)
    g = TaskGrid(p)
    capable = g.rank < 0.5
    with pytest.raises(Infeasible):
        solve_period(g, p, 1e-3, capable)
    m = solve_market(g, p, ComputeSupply(np.array([1.0]), np.array([1e-3])), capable)
    assert m.r > 1e-3
    assert m.period.K_d <= 1.0 * (1.0 + 1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compute_market.py -q`
Expected: collection error `ImportError: cannot import name 'ComputeSupply' from 'cel'`

- [ ] **Step 3: Create `cel/compute_market.py`**

**File: `cel/compute_market.py` (full contents)**

```python
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
```

- [ ] **Step 4: Rewrite `cel/__init__.py`**

**File: `cel/__init__.py` (full contents)**

```python
"""CEL — Compute–Energy–Labor model. Build steps 0–3 implemented."""
from .params import Params, S0, SCENARIOS, scenario, load_params
from .grid import TaskGrid
from .economy import solve_period, labor_share_closed_form, Infeasible, PeriodResult
from .cohorts import DisplacedPool
from .capability import omega_train, omega_inf
from .compute_market import ComputeSupply, MarketResult, MarketError, solve_market
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `72 passed`

- [ ] **Step 6: Amend spec §9.2**

In §9.2, replace everything from the paragraph beginning `Within a period, compute supply is the **vintage merit order**` through the bullet `- inner in r: K_d(w, r) = C_inf (monotone decreasing in r).` with:

```text
Within a period, compute supply is the **vintage merit order**: vintages sorted by operating (energy) cost. The marginal vintage is the dearest one dispatched at price r. If r equals its operating cost the regime is **slack**; if r exceeds it the regime is **scarce**, every dispatched vintage runs at capacity, dearer vintages may sit idle, and the premium r − op_marginal is a scarcity rent (to λ_P if power binds, otherwise to compute owners).

Nested root-finding:
- outer in r: bisection in log r for the smallest r with K_d(r) ≤ S(r). Both curves are step functions (task flips, vintages switching on), so bisection is used instead of Brent;
- inner in w: `brentq` in log w on log(P_T / MPT) = 0, which is increasing in w.

`Infeasible` from the inner solve is treated as unbounded compute demand, so a full run never raises it.
```

Leave the following sentence about the chunking optimum unchanged.

- [ ] **Step 7: Rewrite `cel/simulate.py`**

**File: `cel/simulate.py` (full contents)**

```python
"""
Simulation driver — SKELETON. Per-period update order from spec §11.1.

Build sequence (spec §11.3; each gated by a limiting-case test in tests/):
  [x] 0.  fixes: units, CSV provenance, queue clearing function,
          automatability flag, Omega paths
  [x] 1-2. capital nest                       -> no-AI s_L = 1 - alpha_K; Prop 3 x (1 - alpha_K)
  [x] 3.  merit-order compute market          -> regime switch at K_d = C_inf
  [ ] 4.  vintage capital, scrapping          -> Proposition 1, Hall limit        (Plan 2)
  [ ] 5.  capability and power                -> Propositions 2 and 2b            (Plan 2)
  [ ] 6.  chunking optimization               -> corners under extreme lam        (Plan 3)
  [ ] 7.  queue, scarring, wage floor, fiscal -> Proposition 5; W continuous      (Plan 3)
  [ ] 8.  industry and financing              -> Proposition 4 limits             (Plan 4)
  [ ] 9.  ledger and no-AI counterfactual     -> phi = 0 gives NSV = 0            (Plan 4)
  [ ] 10. calibration, backcast, sensitivity  -> NROY non-empty; out-of-sample    (Plan 5)
"""
from __future__ import annotations
from .params import Params
from .grid import TaskGrid


def run(p: Params):
    grid = TaskGrid(p)
    raise NotImplementedError("steps 4-10 not yet built; see tests/ for steps 0-3")
```

- [ ] **Step 8: Rewrite `README.md`**

**File: `README.md` (full contents)**

```text
# CEL — Compute–Energy–Labor model (build steps 0–3)

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest tests -q

Design spec: docs/superpowers/specs/2026-09-15-cel-model-design.md

Layout
    calibration/parameters.csv  every parameter: value, range, prior, class, source, grade
    cel/params.py          frozen Params dataclass; S0 loaded from the CSV; scenario diffs
    cel/grid.py            Sobol task grid over (d, kappa_E, Gamma, a), equal-measure weights
    cel/economy.py         static period economy: capital nest, task prices, wage solve, factor demands
    cel/compute_market.py  merit-order compute supply and the market-clearing compute price
    cel/capability.py      algorithmic progress paths Omega(t), Omega_inf(t)
    cel/cohorts.py         displaced-worker cohort queue with a clearing function and scarring
    cel/simulate.py        driver skeleton with the step 4–10 checklist
    tests/                 limiting-case tests that gate each build step

Things the tests taught us:
  * With alpha_K = 0 and sigma > 1 the numeraire fails once automated tasks
    alone price the task aggregate below A_Y. With alpha_K > 0 a wage always
    exists, and the compute market treats Infeasible as unbounded demand, so
    a full run never raises it.
  * The old inflow-ratio congestion factor switched off at saturation
    (duration 2000 yr at u = 0.9995, 1 yr at u = 1). The clearing function
    makes duration continuous and monotone; steady state W = c2/(h0(1-u)).
  * Near saturation the queue converges slowly (rate 1 - dX/dS): Kingman
    checks need ~5000 steps at u = 0.95.
  * Negative duration dependence (xi > 0) lowers the employable stock and
    amplifies congestion beyond the Kingman value.
  * With symmetric tasks (theta = 0) a whole block flips at one price, so
    compute demand jumps across capacity; heterogeneous tasks clear to ~1.
  * Forcing automation at an AI price above the wage lowers the wage once
    conventional capital is in the nest: automation must pass the cost test.
```

- [ ] **Step 9: Run the full suite once more**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `72 passed`

- [ ] **Step 10: Commit**

```bash
git add cel/compute_market.py cel/__init__.py cel/simulate.py README.md tests/test_compute_market.py docs/superpowers/specs/2026-09-15-cel-model-design.md
git commit -m "feat: merit-order compute market with endogenous compute price (build step 3)"
```
