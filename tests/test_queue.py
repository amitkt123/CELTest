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
