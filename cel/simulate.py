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
