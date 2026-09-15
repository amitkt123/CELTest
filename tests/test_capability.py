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
