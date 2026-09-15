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
