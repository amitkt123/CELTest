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
