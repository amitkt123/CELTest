"""
Limiting-case tests that gate build steps 1 and 2.

Step 1: no AI (phi_cap = 0). Closed forms with N = L_bar = 1:
    w = A_Y,  Y = A_Y,  s_L = 1,  K_d = 0
Step 2: exogenous phi at fixed r, symmetric tasks (theta = 0 so every
automated task has the same AI price p_A). Proposition 3 must hold
exactly, and its two limits must have the right sign in sigma.
"""
import numpy as np
import pytest

from cel import Params, S0, TaskGrid, solve_period, labor_share_closed_form, Infeasible, DisplacedPool


# ---------------------------------------------------------------- grid sanity
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


# ---------------------------------------------------------------- step 1
@pytest.mark.parametrize("A_Y", [1.0, 2.5])
@pytest.mark.parametrize("sigma", [0.6, 1.5])
def test_step1_no_ai_closed_form(A_Y, sigma):
    p = S0.with_(A_Y=A_Y, sigma=sigma)
    g = TaskGrid(p)
    res = solve_period(g, p, r=1.0, phi_cap=0.0)
    assert res.phi_eff == 0.0 and res.K_d == 0.0
    assert res.w == pytest.approx(A_Y, rel=1e-8)
    assert res.Y == pytest.approx(A_Y, rel=1e-8)
    assert res.L_d == pytest.approx(p.L_bar, rel=1e-8)
    assert res.s_L == pytest.approx(1.0, rel=1e-8)
    assert abs(res.P_Y - 1.0) < 1e-8
    assert abs(res.nominal_check) < 1e-8          # sum p y = P_Y * Y


# ---------------------------------------------------------------- step 2
def _symmetric(sigma, phi, p_A, A_Y=1.0):
    # theta = 0 -> e(i) = e0 for all i; set e0 = 1 so c_AI = r = p_A
    p = S0.with_(sigma=sigma, theta=0.0, e0=1.0, A_Y=A_Y)
    g = TaskGrid(p)
    res = solve_period(g, p, r=p_A, phi_cap=phi, force=True)
    return p, res


@pytest.mark.parametrize("sigma", [0.4, 0.6, 0.9, 1.5])
@pytest.mark.parametrize("phi", [0.1, 0.5, 0.9])
def test_step2_proposition3_exact(sigma, phi):
    p_A = 0.9                       # inside feasibility bound phi*p_A^(1-sigma) < 1 for all cases
    p, res = _symmetric(sigma, phi, p_A)
    assert res.phi_eff == pytest.approx(phi, abs=1.0 / p.M_tasks)
    assert np.all(res.prices[res.auto] == p_A)
    # cost condition should hold at the solution even though we forced it
    assert p_A < res.w
    expected = labor_share_closed_form(p, res.phi_eff, res.w, p_A)
    assert res.s_L == pytest.approx(expected, rel=1e-8)
    assert abs(res.nominal_check) < 1e-8
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
    # below the feasibility boundary phi * p_A^(1-sigma) = A_Y^(1-sigma)
    # the numeraire cannot be sustained: w -> infinity, s_L -> 0
    with pytest.raises(Infeasible):
        _symmetric(1.5, phi, 0.2)


def test_step2_deployment_wedge_when_not_forced():
    # theta > 0: hard tasks are expensive to run, so phi_eff < phi_cap
    p = S0.with_(sigma=0.6, theta=1.0, e0=1.0)
    g = TaskGrid(p)
    res = solve_period(g, p, r=0.3, phi_cap=0.9, force=False)
    assert 0.0 < res.phi_eff < 0.9
    assert np.all(res.prices[res.auto] < res.w / g.gamma[res.auto])


# ---------------------------------------------------------------- cohort queue
def _pure_queue():
    # no scarring exits, no duration dependence: isolates Kingman congestion
    return S0.with_(pi_scar0=0.0, pi_scar_slope=0.0, xi_reemp=0.0, c2_burst=1.0)


def test_queue_kingman_exact_without_duration_dependence():
    # steady-state duration = 1/h_eff = 1 + c2 * u/(1-u), with h0 = 1
    for inflow, expected in [(0.005, 1 + 1/3), (0.010, 2.0), (0.015, 4.0), (0.019, 20.0)]:
        q = DisplacedPool(_pure_queue())
        for _ in range(400):
            st = q.step(inflow=inflow, capacity=0.02)
        assert st.mean_duration == pytest.approx(expected, rel=0.02), (inflow, st.mean_duration)


def test_queue_saturated_grows_without_bound():
    q = DisplacedPool(_pure_queue())
    stocks = [q.step(inflow=0.03, capacity=0.02).stock for _ in range(40)]
    assert stocks[-1] > stocks[-10] > stocks[-20]
    assert q.step(0.03, 0.02).mean_duration > 10.0


def test_queue_duration_dependence_amplifies_congestion():
    # with xi > 0 the congested pool ages into low-hazard cohorts: duration
    # exceeds the Kingman value. This is negative duration dependence x
    # congestion, a documented amplification, and it is a model output.
    dep = S0.with_(pi_scar0=0.0, pi_scar_slope=0.0, xi_reemp=0.35, c2_burst=1.0)
    q = DisplacedPool(dep)
    for _ in range(200):
        st = q.step(inflow=0.010, capacity=0.02)
    assert st.mean_duration > 2.0            # > Kingman value at u = 0.5
