"""Reduced reproduction of the Miller, Remer, Ryan and Sheu (2017) Monte Carlo."""

import numpy as np
import pytest

from mergerlab.mrrs import simulate_markets


@pytest.fixture(scope="module")
def draws():
    return simulate_markets(1500, seed=2017)


def test_order_statistics_match_published_medians(draws):
    """Published (Table 1): HHI 1,562 to 1,931, delta 317, UPP 0.07, logit 0.06, linear 0.05."""
    assert np.median(draws.hhi_pre) == pytest.approx(1562, rel=0.04)
    assert np.median(draws.hhi_post) == pytest.approx(1931, rel=0.04)
    assert np.median(draws.delta_hhi) == pytest.approx(317, rel=0.12)
    assert np.median(draws.upp) == pytest.approx(0.07, abs=0.01)
    assert np.median(draws.effect_logit) == pytest.approx(0.06, abs=0.01)
    assert np.median(draws.effect_linear) == pytest.approx(0.05, abs=0.01)
    assert np.median(draws.margin1) == pytest.approx(0.49, abs=0.03)
    assert np.median(draws.share1) == pytest.approx(0.15, abs=0.02)


def test_upp_is_diversion_times_margin_and_first_order_is_accurate(draws):
    assert np.all(draws.upp > 0)
    assert np.corrcoef(draws.upp, draws.effect_logit)[0, 1] > 0.99  # paper: 0.996
    err_upp = np.median(np.abs(draws.upp - draws.effect_logit))
    err_foa = np.median(np.abs(draws.foa_logit - draws.effect_logit))
    assert err_upp < 0.012  # paper, Table 2: 0.006
    assert err_foa < err_upp
    assert np.median(draws.effect_linear) < np.median(draws.effect_logit)


def test_design_is_reproducible_and_reports_redraws():
    a = simulate_markets(50, seed=3)
    b = simulate_markets(50, seed=3)
    np.testing.assert_array_equal(a.effect_logit, b.effect_logit)
    assert a.redraws >= 0 and a.solver_failures == 0
    assert len(a) == 50
