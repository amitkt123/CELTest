# CEL — Compute–Energy–Labor model (build steps 1–2)

    pip install numpy scipy pytest
    python -m pytest tests -q

Layout
    cel/params.py    frozen Params dataclass, S0 baseline, scenario diffs
    cel/grid.py      Sobol task grid over (d, kappa_E, Gamma), equal-measure weights
    cel/economy.py   static period economy: task prices, P_Y = 1 bisection, factor demands
    cel/cohorts.py   displaced-worker cohort queue with Kingman congestion + scarring
    cel/simulate.py  driver skeleton with the step 3–8 checklist
    tests/           limiting-case tests that gate each build step

Things the tests already taught us (see test comments):
  * With sigma > 1 the numeraire P_Y = 1 is infeasible once
    phi * p_A^(1-sigma) > A_Y^(1-sigma). Not a bug: r is endogenous and
    bounded below by energy cost in the dynamic model.
  * A deterministic fluid queue has no congestion below capacity. The
    (mu - lambda)^-1 explosion is stochastic; it enters via Kingman's
    formula with burstiness c2, which cascades raise.
  * Negative duration dependence (xi > 0) amplifies congestion well beyond
    the Kingman value. Magnitude is a model output; keep it in view.
