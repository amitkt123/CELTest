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
