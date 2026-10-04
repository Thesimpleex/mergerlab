"""Reference values from the CRAN package ``antitrust`` (see scripts/oracles)."""

import numpy as np
import pytest

from helpers import nan_list
from mergerlab.demand import (
    calibrate_linear,
    calibrate_logit,
    calibrate_logit_alm,
    calibrate_pcaids,
)
from mergerlab.market import Market
from mergerlab.metrics import (
    cmcr_from_demand,
    cmcr_from_diversion,
    compensating_variation,
    diversion_from_demand,
)
from mergerlab.screens import concentration
from mergerlab.supply import solve_bertrand
from mergerlab.units import Diversion, DiversionBasis, Margins, Ownership, Shares


def _pct(post, pre):
    return (post / pre - 1.0) * 100.0


def test_fixture_records_pinned_versions(r_fixtures):
    assert r_fixtures["antitrust_version"] == "0.99.33"
    assert r_fixtures["r_version"].startswith("R version")


def test_pcaids_epstein_rubinfeld_example(r_fixtures):
    f = r_fixtures["pcaids_epstein_rubinfeld"]
    p = np.array(f["prices"])
    mk = Market.build(
        p, Shares.within(f["revenue_shares"], "revenue"), Margins.lerner([np.nan] * 3), list("ABC")
    )
    cal = calibrate_pcaids(mk, f["own_elasticity_first"], f["market_elasticity"])
    post = mk.ownership.merged(["A", "B"])
    eq = solve_bertrand(cal.demand, cal.costs, post, p)
    np.testing.assert_allclose(_pct(eq.prices, p), f["price_change_pct"], rtol=1e-6)
    np.testing.assert_allclose(_pct(eq.prices, p), [13.7639, 10.7539, 4.0596], atol=5e-5)
    cm = cmcr_from_demand(cal.demand, p, cal.costs, post)
    np.testing.assert_allclose(100 * cm.relative[:2], f["cmcr_pct"], rtol=1e-10)
    np.testing.assert_allclose(100 * cm.relative[:2], [16.667, 12.698], atol=5e-4)
    c = concentration(mk.shares, mk.ownership, post)
    assert (c.hhi_pre, c.hhi_post) == (pytest.approx(f["hhi_pre"]), pytest.approx(f["hhi_post"]))


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_pcaids_away_from_market_elasticity_minus_one_matches_r(r_fixtures, case_index):
    """Expenditure equation and (eps_m + 1) terms: price effects, margins, elasticities, CMCR."""
    f = r_fixtures["pcaids_market_elasticity"]
    case = f["cases"][case_index]
    p = np.array(f["prices"])
    w = np.array(f["revenue_shares"])
    mk = Market.build(p, Shares.within(w, "revenue"), Margins.lerner([np.nan] * 3), list("ABC"))
    cal = calibrate_pcaids(mk, f["own_elasticity_first"], case["market_elasticity"])
    post = mk.ownership.merged(["A", "B"])
    eq = solve_bertrand(cal.demand, cal.costs, post, p)
    np.testing.assert_allclose(_pct(eq.prices, p), case["price_change_pct"], rtol=1e-6)
    np.testing.assert_allclose(cal.fitted_margins, case["margins"], rtol=1e-10)
    np.testing.assert_allclose(cal.demand.elasticities(p), case["elasticities"], rtol=1e-10)
    cm = cmcr_from_demand(cal.demand, p, cal.costs, post)
    np.testing.assert_allclose(100 * cm.relative[:2], case["cmcr_pct"], rtol=1e-10)


@pytest.mark.parametrize("case_index", [0, 1, 2])
def test_r_aids_share_diversion_converts_to_the_cmcr_r_reports(r_fixtures, case_index):
    """R's AIDS diversion is a share-slope ratio; only with the market elasticity does it give
    the quantity diversion that reproduces R's own CMCR. Treating it as a revenue-change ratio
    is right only at a market elasticity of -1."""
    f = r_fixtures["pcaids_market_elasticity"]
    case = f["cases"][case_index]
    eps_m = case["market_elasticity"]
    p = np.array(f["prices"])
    w = np.array(f["revenue_shares"])
    mk = Market.build(p, Shares.within(w, "revenue"), Margins.lerner([np.nan] * 3), list("ABC"))
    post = mk.ownership.merged(["A", "B"])
    margins = Margins.lerner(case["margins"])
    cal = calibrate_pcaids(mk, f["own_elasticity_first"], eps_m)
    own_elasticity = np.diag(np.array(case["elasticities"]))
    r_div = Diversion.share(np.array(case["share_diversion"]))
    from_demand = diversion_from_demand(cal.demand, p, DiversionBasis.SHARE)
    np.testing.assert_allclose(
        from_demand.matrix, np.where(np.eye(3, dtype=bool), 0.0, r_div.matrix), rtol=1e-10
    )
    quantity = r_div.to_quantity(p, own_elasticity, shares=w, market_elasticity=eps_m)
    cm = cmcr_from_diversion(p, margins, quantity, mk.ownership, post)
    np.testing.assert_allclose(100 * cm.relative[:2], case["cmcr_pct"], rtol=1e-10)
    as_revenue = Diversion.revenue(np.array(case["share_diversion"])).to_quantity(p, own_elasticity)
    wrong = cmcr_from_diversion(p, margins, as_revenue, mk.ownership, post)
    assert np.max(np.abs(100 * wrong.relative[:2] - case["cmcr_pct"])) > 0.5


def test_werden_table_1_grid_matches_r_and_closed_form(r_fixtures):
    from mergerlab.metrics import cmcr_two_product

    f = r_fixtures["cmcr_werden_table1"]
    grid = np.array(f["cmcr_pct"])
    for i, d in enumerate(f["diversions"]):
        for j, m in enumerate(f["margins"]):
            closed = 100 * cmcr_two_product(m, m, d, d, 1.0)
            assert closed == pytest.approx(grid[i, j], rel=1e-12)
    assert grid.min() == pytest.approx(3.5088, abs=1e-4)
    assert grid.max() == pytest.approx(77.7778, abs=1e-4)


def test_cmcr_bertrand_documentation_example(r_fixtures):
    f = r_fixtures["cmcr_bertrand_doc"]
    p = np.array(f["prices"])
    # R's ownerPre = c(1, 0, 0): product 1 belongs to one party, products 2 and 3 to the
    # other; the default ownerPost is the all-ones matrix (one merged firm).
    own_pre = Ownership.from_owners(["M", "N", "N"])
    own_post = Ownership(np.ones((3, 3)))
    res = cmcr_from_diversion(
        p, Margins.lerner(f["margins"]), Diversion.quantity(f["diversion"]), own_pre, own_post
    )
    np.testing.assert_allclose(100 * res.relative, f["cmcr_pct"], rtol=1e-9)


def test_logit_known_outside_share(r_fixtures):
    f = r_fixtures["logit_known_outside"]
    p = np.array(f["prices"])
    q = np.array(f["quantity_shares"])
    mk = Market.build(
        p,
        Shares.total(q, "quantity"),
        Margins.lerner(nan_list(f["margins"])),
        f["owner_pre"],
        revenue=float(p @ q),
    )
    cal = calibrate_logit(mk)
    assert cal.parameters["alpha"] == pytest.approx(f["alpha"], rel=2e-6)
    post = Ownership.from_owners(f["owner_post"])
    eq = solve_bertrand(cal.demand, cal.costs, post, p)
    np.testing.assert_allclose(_pct(eq.prices, p), f["price_change_pct"], rtol=1e-5)
    cv = compensating_variation(cal.demand, p, eq.prices)
    assert cv == pytest.approx(f["compensating_variation"], rel=1e-5)


def test_logit_with_multiproduct_firm_pre_merger(r_fixtures):
    f = r_fixtures["logit_multiproduct"]
    p = np.array(f["prices"])
    mk = Market.build(
        p,
        Shares.total(f["quantity_shares"], "quantity"),
        Margins.lerner(nan_list(f["margins"])),
        f["owner_pre"],
    )
    cal = calibrate_logit(mk)
    assert cal.parameters["alpha"] == pytest.approx(f["alpha"], rel=2e-6)
    eq = solve_bertrand(cal.demand, cal.costs, Ownership.from_owners(f["owner_post"]), p)
    np.testing.assert_allclose(_pct(eq.prices, p), f["price_change_pct"], rtol=1e-5)


def test_logit_alm_exactly_identified(r_fixtures):
    f = r_fixtures["logit_alm_exact"]
    p = np.array(f["prices"])
    mk = Market.build(
        p,
        Shares.within(f["inside_shares"], "quantity"),
        Margins.lerner(nan_list(f["margins"])),
        f["owner_pre"],
    )
    cal = calibrate_logit_alm(mk)
    assert cal.parameters["alpha"] == pytest.approx(f["alpha_true"], rel=1e-9)
    assert cal.parameters["outside_share"] == pytest.approx(f["outside_share_true"], rel=1e-9)
    assert cal.parameters["alpha"] == pytest.approx(f["alpha"], rel=5e-6)
    assert 1 - cal.parameters["outside_share"] == pytest.approx(f["shareInside"], rel=5e-6)
    eq = solve_bertrand(cal.demand, cal.costs, Ownership.from_owners(f["owner_post"]), p)
    np.testing.assert_allclose(_pct(eq.prices, p), f["price_change_pct"], rtol=1e-5)


def test_linear_demand_with_given_diversion(r_fixtures):
    f = r_fixtures["linear_given_diversion"]
    p = np.array(f["prices"])
    q = np.array(f["quantities"])
    d = np.array(f["diversion"])
    mk = Market.build(
        p,
        Shares.within(q / q.sum(), "quantity"),
        Margins.lerner(f["margins"]),
        f["owner_pre"],
        revenue=float(p @ q),
    )
    cal = calibrate_linear(mk, Diversion.quantity(d))
    np.testing.assert_allclose(cal.costs, f["marginal_costs"], rtol=1e-12)
    np.testing.assert_allclose(cal.demand.slopes, np.array(f["slopes"]), rtol=1e-10)
    eq = solve_bertrand(cal.demand, cal.costs, Ownership.from_owners(f["owner_post"]), p)
    np.testing.assert_allclose(_pct(eq.prices, p), f["price_change_pct"], rtol=1e-8)
