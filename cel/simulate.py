"""
Simulation driver — SKELETON. Per-period update order from spec §11.

Build sequence (each gated by a limiting-case test in tests/):
  [x] 1. task economy, phi = 0                       -> Y = A_Y, s_L = 1
  [x] 2. exogenous phi at fixed r                    -> Proposition 3 closed form
  [ ] 3. nested bisection (w outer, r inner)         -> regime switch at K_d = C_inf
  [ ] 4. vintage capital, scrapping, delta_econ      -> Proposition 1
  [ ] 5. capability block (Omega, x(t), H)           -> Proposition 2 ceiling
  [ ] 6. chunking optimization -> omega, phi jumps   -> corners under extreme lam
  [ ] 7. queue, scarring, wage floor, fiscal b(t)    -> duration explodes near saturation
  [ ] 8. ledger: NSV, NPV_lab, dW                    -> S0 backcast vs 2023-26 series
"""
from __future__ import annotations
from .params import Params
from .grid import TaskGrid


def run(p: Params):
    grid = TaskGrid(p)
    raise NotImplementedError("steps 3-8 not yet built; see tests/ for steps 1-2")
