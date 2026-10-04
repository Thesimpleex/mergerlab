"""Concentration-based consumer harm (Koh 2025): Table 1 and algebraic identities."""

import numpy as np
import pytest

from mergerlab import (
    CES,
    Linear,
    Logit,
    Margins,
    Market,
    Shares,
    calibrate_ces,
    first_order_harm_error,
    harm_comparison,
    koh_decomposition,
    koh_rho1,
    recover_costs,
)
from mergerlab.units import Ownership

SHARES = [0.174, 0.154, 0.65]  # Heinz, Beech-Nut, Gerber (revenue shares of the market)
OUTSIDE = 0.022
INCOME = 865.0

# Koh (2025), Table 1: sigma, phi, V0, rho1, rho2, delta CS in million USD (printed precision).
TABLE_1 = [
    (1.5, 3.00, 1730.00, 0.37, 1.09, 37.68),
    (2.0, 2.00, 865.00, 0.59, 1.04, 28.48),
    (2.5, 1.67, 576.67, 0.73, 1.00, 22.87),
    (3.0, 1.50, 432.50, 0.84, 0.98, 19.10),
]


def heinz_ces(sigma: float):
    """CES demand whose Bertrand margins at these shares follow from ``sigma``."""
    w = np.array(SHARES)
    margin = 1.0 / (sigma - (sigma - 1.0) * w[0])
    market = Market.build(
        [1.0, 1.0, 1.0],
        Shares.total(w, "revenue", OUTSIDE),
        Margins.lerner([margin, np.nan, np.nan]),
        ["Heinz", "Beech-Nut", "Gerber"],
        revenue=INCOME * (1 - OUTSIDE),
    )
    return calibrate_ces(market)


@pytest.mark.parametrize(("sigma", "phi", "v0", "rho1", "rho2", "harm"), TABLE_1)
def test_koh_table_1_heinz_beech_nut(sigma, phi, v0, rho1, rho2, harm):
    cal = heinz_ces(sigma)
    assert cal.parameters["sigma"] == pytest.approx(sigma, rel=1e-8)
    dec = koh_decomposition(cal.demand, np.ones(3), cal.costs, 0, 1)
    assert dec.phi == pytest.approx(phi, abs=5e-3)
    assert dec.v0 == pytest.approx(v0, abs=5e-3)
    assert dec.rho1 == pytest.approx(rho1, abs=1e-2)  # the table truncates 0.738 to 0.73
    assert dec.rho2 == pytest.approx(rho2, abs=6e-3)
    assert dec.delta_hhi == pytest.approx(0.0536, abs=5e-5)
    assert dec.harm == pytest.approx(harm, abs=0.006)
    assert dec.harm == pytest.approx(dec.rho * dec.delta_hhi)


def test_koh_rho1_limits_and_validation():
    assert koh_rho1(1.0, 1e-6, 1e-6) == pytest.approx(1.0, abs=1e-5)
    assert koh_rho1(3.0, 1e-6, 1e-6) == pytest.approx(1 / 3, abs=1e-5)
    assert koh_rho1(1.0, 0.3, 0.2) > koh_rho1(1.0, 0.25, 0.25) * 0.99  # asymmetry raises rho1
    with pytest.raises(ValueError, match="smaller than phi"):
        koh_rho1(1.0, 1.0, 0.2)


def _logit_market(shares, alpha=2.0, size=1000.0):
    s = np.asarray(shares)
    delta = np.log(s / (1 - s.sum())) + alpha  # unit prices
    demand = Logit(delta, alpha, size)
    costs = recover_costs(demand, np.ones(s.size), Ownership.from_owners([f"F{i}" for i in s]))
    return demand, costs


def test_logit_decomposition_matches_closed_form_pass_through_expression():
    """rho2 from the definition equals Koh's expression in the elements of M (unit prices)."""
    demand, costs = _logit_market([0.2, 0.1, 0.25, 0.15])
    for rivals in ("fixed", "respond"):
        dec = koh_decomposition(demand, np.ones(4), costs, 0, 1, rivals)
        m = dec.pass_through
        sa, sb = dec.share_a, dec.share_b
        closed = (m[0, 0] + m[1, 1] + m[0, 1] * sa / sb + m[1, 0] * sb / sa) / (2 * dec.phi)
        assert dec.rho2 == pytest.approx(closed, rel=1e-9)


def test_logit_closed_form_pricing_pressure_and_small_share_limit():
    """With M = I and unit prices the harm is (N/alpha) dHHI / ((1-sA)(1-sB))."""
    demand, costs = _logit_market([2e-4, 3e-4, 0.3])
    dec = koh_decomposition(demand, np.ones(3), costs, 0, 1)
    assert dec.rho2 == pytest.approx(1.0, abs=1e-3)  # Proposition 3
    expected = demand.market_size / demand.alpha * dec.delta_hhi / ((1 - 2e-4) * (1 - 3e-4))
    assert dec.harm == pytest.approx(expected, rel=1e-3)
    assert dec.harm_with_rho2_one == pytest.approx(expected, rel=1e-12)


def test_first_order_matches_simulation_for_small_mergers_and_large_rivals_respond():
    small, c_small = _logit_market([0.01, 0.015, 0.02])
    fo, sim = first_order_harm_error(small, np.ones(3), c_small, 0, 1)
    assert fo == pytest.approx(sim, rel=0.01)
    # a large rival re-prices too; Koh's formula counts only the merging firms' products
    big_rival, c_big = _logit_market([0.01, 0.015, 0.3])
    fo, sim = first_order_harm_error(big_rival, np.ones(3), c_big, 0, 1)
    assert sim > 1.05 * fo


@pytest.mark.parametrize(
    ("sigma", "ratio"), [(1.5, 0.941), (2.0, 0.977), (2.5, 0.999), (3.0, 1.013)]
)
def test_first_order_formula_is_accurate_like_for_like_and_the_gap_is_the_rival_response(
    sigma, ratio
):
    """Koh's formula covers the merging products with rivals' prices fixed. Against the exact
    compensating variation of that same price change it is within 6% (CES curvature); the
    full simulation adds Gerber's price response, which is the larger part of the gap."""
    cal = heinz_ces(sigma)
    cmp = harm_comparison(cal.demand, np.ones(3), cal.costs, 0, 1)
    assert cmp.first_order / cmp.exact_rivals_fixed == pytest.approx(ratio, abs=2e-3)
    assert abs(cmp.curvature_gap) < 0.2 * cmp.rival_response
    assert cmp.rival_response_share > 0.30
    assert cmp.full_simulation > 1.5 * cmp.first_order
    assert cmp.price_change_rivals_fixed[2] == 0.0
    assert cmp.price_change_full[2] > 0.02  # Gerber raises its price after the merger
    assert cmp.full_simulation == pytest.approx(
        cmp.first_order + cmp.curvature_gap + cmp.rival_response
    )


def test_koh_formula_equals_exact_harm_when_rivals_cannot_matter():
    """With a negligible outside rival and tiny shares the three harms coincide."""
    demand, costs = _logit_market([2e-4, 3e-4, 1e-4])
    cmp = harm_comparison(demand, np.ones(3), costs, 0, 1)
    assert cmp.first_order == pytest.approx(cmp.exact_rivals_fixed, rel=1e-3)
    assert cmp.full_simulation == pytest.approx(cmp.exact_rivals_fixed, rel=1e-3)


def test_decomposition_input_validation():
    demand, costs = _logit_market([0.2, 0.1, 0.25])
    with pytest.raises(ValueError, match="two distinct"):
        koh_decomposition(demand, np.ones(3), costs, 0, 0)
    with pytest.raises(ValueError, match="rivals"):
        koh_decomposition(demand, np.ones(3), costs, 0, 1, rivals="both")
    ces = CES(np.array([0.5, 0.5]), 2.0, 10.0, np.ones(2))
    assert ces.n_products == 2
    linear = Linear(np.ones(2), -np.eye(2), np.ones(2))
    with pytest.raises(ValueError, match="logit and CES"):
        koh_decomposition(linear, np.ones(2), np.zeros(2), 0, 1)
