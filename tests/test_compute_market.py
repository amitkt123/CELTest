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
