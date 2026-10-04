import numpy as np
import pytest
from scipy.special import lambertw

from helpers import numerical_jacobian
from mergerlab.demand import (
    CES,
    PCAIDS,
    CalibrationError,
    Linear,
    Logit,
    NestedLogit,
    calibrate_ces,
    calibrate_linear,
    calibrate_logit,
    calibrate_logit_alm,
    calibrate_nested_logit,
    calibrate_pcaids,
)
from mergerlab.market import Market
from mergerlab.supply import solve_bertrand
from mergerlab.units import Diversion, Margins, Shares

P4 = np.array([1.0, 1.3, 0.9, 1.1])
OWNERS4 = ["A", "B", "C", "D"]


def models():
    rng = np.random.default_rng(3)
    delta = rng.uniform(0.5, 2.0, 4)
    return [
        Logit(delta, 1.7, market_size=3.0),
        NestedLogit(delta, 1.7, 0.45, [0, 0, 1, 1], market_size=3.0),
        CES(rng.uniform(0.2, 1.0, 4), 2.6, 5.0, P4),
        PCAIDS(
            np.array([0.1, 0.2, 0.3, 0.4]),
            -2.0
            * (
                np.diag([0.1, 0.2, 0.3, 0.4]) - np.outer([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4])
            ),
            -0.7,
            5.0,
            P4,
        ),
    ]


@pytest.mark.parametrize("model", models(), ids=lambda m: m.name)
def test_jacobian_matches_finite_differences(model):
    p = P4 * np.array([1.02, 0.97, 1.01, 1.04])
    num = numerical_jacobian(model.quantities, p)
    np.testing.assert_allclose(model.jacobian(p), num, rtol=1e-6, atol=1e-8)


def test_linear_jacobian_is_constant():
    b = np.array([[-2.0, 0.5], [0.4, -1.5]])
    lin = Linear(np.array([3.0, 2.0]), b, np.ones(2))
    np.testing.assert_array_equal(lin.jacobian(np.array([1.0, 2.0])), b)


@pytest.mark.parametrize("model", models()[:2], ids=lambda m: m.name)
def test_logit_family_surplus_equals_path_integral(model):
    """Logit and nested logit have no income effects: CS change = -integral of q dp."""
    p0, p1 = P4, P4 * np.array([1.1, 1.05, 0.95, 1.2])
    nodes, weights = np.polynomial.legendre.leggauss(30)
    t, w = 0.5 * (nodes + 1), 0.5 * weights
    integral = sum(
        wi * model.quantities(p0 + ti * (p1 - p0)) @ (p1 - p0) for ti, wi in zip(t, w, strict=True)
    )
    assert model.consumer_surplus(p1) - model.consumer_surplus(p0) == pytest.approx(
        -integral, rel=1e-12
    )


def test_linear_surplus_equals_path_integral_for_symmetric_slopes():
    """Linear demand with symmetric slopes: CS(p1) - CS(p0) = -integral of q dp from any p0."""
    slopes = np.array([[-2.0, 0.6, 0.3], [0.6, -1.5, 0.4], [0.3, 0.4, -1.8]])
    ref = np.array([1.0, 1.2, 0.9])
    model = Linear(np.array([4.0, 3.0, 3.5]) - slopes @ ref, slopes, ref)
    p0, p1 = np.array([1.1, 1.0, 0.8]), np.array([1.4, 1.3, 1.0])
    nodes, weights = np.polynomial.legendre.leggauss(8)
    t, w = 0.5 * (nodes + 1), 0.5 * weights
    integral = sum(
        wi * model.quantities(p0 + ti * (p1 - p0)) @ (p1 - p0) for ti, wi in zip(t, w, strict=True)
    )
    assert model.consumer_surplus(p1) - model.consumer_surplus(p0) == pytest.approx(
        -integral, rel=1e-12
    )


def test_linear_surplus_single_product_is_the_demand_triangle():
    """q = 10 - 2p: raising the price from 1 to 2 costs the integral of (10 - 2p), which is 7."""
    model = Linear(np.array([10.0]), np.array([[-2.0]]), np.array([1.0]))
    change = model.consumer_surplus(np.array([2.0])) - model.consumer_surplus(np.array([1.0]))
    assert change == pytest.approx(-7.0, rel=1e-14)


def test_nested_logit_with_zero_sigma_is_logit():
    delta = np.array([1.0, 1.5, 0.7])
    p = np.array([1.0, 1.2, 0.9])
    a = Logit(delta, 2.0, 2.0)
    b = NestedLogit(delta, 2.0, 0.0, [0, 0, 1], 2.0)
    np.testing.assert_allclose(a.quantities(p), b.quantities(p), rtol=1e-14)
    np.testing.assert_allclose(a.jacobian(p), b.jacobian(p), rtol=1e-13)
    assert a.consumer_surplus(p) == pytest.approx(b.consumer_surplus(p), rel=1e-14)


def test_ces_compensating_variation_matches_price_index():
    """CV = Y (P'/P - 1) with the CES price index of the outside-good formulation."""
    beta = np.array([0.4, 0.7, 0.3])
    sigma, income = 3.1, 7.0
    p0 = np.array([1.0, 1.2, 0.8])
    p1 = np.array([1.3, 1.1, 0.9])
    model = CES(beta, sigma, income, p0)
    index = lambda p: (1.0 + np.sum(beta * p ** (1 - sigma))) ** (1.0 / (1.0 - sigma))  # noqa: E731
    cv_index = income * (index(p1) / index(p0) - 1.0)
    cv_model = model.consumer_surplus(p0) - model.consumer_surplus(p1)
    assert cv_model == pytest.approx(cv_index, rel=1e-12)


def test_pcaids_surplus_closed_form_when_market_elasticity_is_minus_one():
    w = np.array([0.2, 0.3, 0.5])
    p0 = np.array([2.9, 3.4, 2.2])
    slopes = -2.5 * (np.diag(w) - np.outer(w, w))
    model = PCAIDS(w, slopes, -1.0, 4.0, p0)
    p1 = p0 * np.array([1.1, 1.05, 1.0])
    x = np.log(p1 / p0)
    expected = -4.0 * (w @ x + 0.5 * x @ slopes @ x)
    assert model.consumer_surplus(p1) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- logit calibration


def logit_market(margins, owners=None, revenue=None):
    return Market.build(
        P4,
        Shares.total([0.15, 0.25, 0.20, 0.10], "quantity"),
        Margins.lerner(margins),
        owners or OWNERS4,
        revenue=revenue,
    )


def test_logit_single_margin_is_exactly_identified():
    cal = calibrate_logit(logit_market([0.4, np.nan, np.nan, np.nan]))
    s = np.array([0.15, 0.25, 0.20, 0.10])
    assert cal.parameters["alpha"] == pytest.approx(1.0 / (0.4 * 1.0 * (1 - 0.15)), rel=1e-14)
    np.testing.assert_allclose(cal.demand.shares(P4), s, rtol=1e-13)
    assert cal.fitted_margins[0] == pytest.approx(0.4, rel=1e-13)
    assert cal.max_margin_residual < 1e-13


def test_logit_common_markup_condition_for_multiproduct_firm():
    cal = calibrate_logit(logit_market([0.4, np.nan, np.nan, np.nan], ["A", "B", "A", "D"]))
    s = cal.demand.shares(P4)
    alpha = cal.parameters["alpha"]
    markups = P4 - cal.costs
    firm_a = s[0] + s[2]
    assert markups[0] == pytest.approx(1.0 / (alpha * (1 - firm_a)), rel=1e-12)
    assert markups[2] == pytest.approx(markups[0], rel=1e-12)
    assert markups[1] == pytest.approx(1.0 / (alpha * (1 - s[1])), rel=1e-12)


def test_logit_least_squares_reports_residuals():
    margins = [0.40, 0.33, 0.45, 0.30]
    cal = calibrate_logit(logit_market(margins))
    assert np.all(np.isfinite(cal.margin_residuals))
    assert cal.max_margin_residual > 1e-3
    np.testing.assert_allclose(
        cal.fitted_margins - np.array(margins), cal.margin_residuals, atol=1e-12
    )


def test_logit_rejects_wrong_share_conventions():
    rev = Market.build(
        P4,
        Shares.total([0.15, 0.25, 0.20, 0.10], "revenue"),
        Margins.lerner([0.4, np.nan, np.nan, np.nan]),
        OWNERS4,
    )
    with pytest.raises(ValueError, match="requires quantity shares"):
        calibrate_logit(rev)
    inside = Market.build(
        P4, Shares.within([0.3, 0.3, 0.2, 0.2], "quantity"), Margins.lerner([0.4] * 4), OWNERS4
    )
    with pytest.raises(ValueError, match="outside good"):
        calibrate_logit(inside)


def test_single_product_logit_monopoly_matches_lambert_w():
    """Closed form p* = c + (1 + W(exp(delta - alpha c - 1))) / alpha."""
    delta, alpha, c = 2.3, 1.4, 0.8
    demand = Logit(np.array([delta]), alpha)
    eq = solve_bertrand(demand, np.array([c]), np.ones((1, 1)), np.array([c + 1.0]))
    closed = c + (1.0 + lambertw(np.exp(delta - alpha * c - 1.0)).real) / alpha
    assert eq.prices[0] == pytest.approx(closed, rel=1e-12)


# ------------------------------------------------------------------------ logit ALM


def test_logit_alm_recovers_alpha_and_outside_share():
    alpha, s_tot = 2.5, np.array([0.15, 0.25, 0.20])
    p = np.array([1.0, 1.3, 0.9])
    margins = 1.0 / (alpha * (1 - s_tot)) / p
    mk = Market.build(
        p,
        Shares.within(s_tot / s_tot.sum(), "quantity"),
        Margins.lerner([margins[0], margins[1], np.nan]),
        ["A", "B", "C"],
    )
    cal = calibrate_logit_alm(mk)
    assert cal.parameters["alpha"] == pytest.approx(alpha, rel=1e-9)
    assert cal.parameters["outside_share"] == pytest.approx(0.4, rel=1e-9)
    assert cal.max_margin_residual < 1e-10


def test_logit_alm_raises_at_the_boundary_instead_of_returning_silently():
    """Epstein-Rubinfeld inputs: margins from PCAIDS push the outside share to 1."""
    p = np.array([2.9, 3.4, 2.2])
    qs = Shares.within([0.2, 0.3, 0.5], "revenue").to_quantity(p)
    mk = Market.build(p, qs, Margins.lerner([1 / 3, 1 / 2.75, 1 / 2.25]), ["A", "B", "C"])
    with pytest.raises(CalibrationError, match=r"outside share is 1\.0000, close to 1"):
        calibrate_logit_alm(mk)


def test_logit_alm_needs_two_margins_with_distinct_firm_shares():
    p = np.ones(3)
    shares = Shares.within([0.2, 0.3, 0.5], "quantity")
    with pytest.raises(CalibrationError, match="at least two margins"):
        calibrate_logit_alm(
            Market.build(p, shares, Margins.lerner([0.3, np.nan, np.nan]), ["A", "B", "C"])
        )
    equal = Shares.within([0.25, 0.25, 0.5], "quantity")
    with pytest.raises(CalibrationError, match="not identified"):
        calibrate_logit_alm(
            Market.build(p, equal, Margins.lerner([0.3, 0.3, np.nan]), ["A", "B", "C"])
        )


# ------------------------------------------------------------------- nested logit


@pytest.mark.parametrize("sigma", [0.0, 0.3, 0.6, 0.9])
def test_nested_logit_calibration_reproduces_shares_and_margin(sigma):
    mk = logit_market([0.4, np.nan, np.nan, np.nan])
    cal = calibrate_nested_logit(mk, [0, 0, 1, 1], sigma)
    np.testing.assert_allclose(cal.demand.shares(P4), [0.15, 0.25, 0.20, 0.10], rtol=1e-12)
    assert cal.fitted_margins[0] == pytest.approx(0.4, rel=1e-12)


def test_nested_logit_sigma_zero_equals_logit_calibration():
    mk = logit_market([0.4, np.nan, np.nan, np.nan], ["A", "B", "A", "D"])
    a = calibrate_logit(mk)
    b = calibrate_nested_logit(mk, [0, 0, 1, 1], 0.0)
    assert b.parameters["alpha"] == pytest.approx(a.parameters["alpha"], rel=1e-13)
    np.testing.assert_allclose(b.costs, a.costs, rtol=1e-12)


def test_nested_logit_rejects_bad_sigma():
    with pytest.raises(ValueError, match="sigma"):
        NestedLogit(np.zeros(2), 1.0, 1.0, [0, 0])


# ---------------------------------------------------------------------------- CES


def test_ces_single_product_lerner_is_closed_form():
    """One single-product firm: m = 1 / (sigma - (sigma - 1) w)."""
    w, sigma = 0.35, 2.4
    mk = Market.build(
        np.array([1.7]),
        Shares.total([w], "revenue"),
        Margins.lerner([1.0 / (sigma - (sigma - 1) * w)]),
        ["A"],
        revenue=3.0,
    )
    cal = calibrate_ces(mk)
    assert cal.parameters["sigma"] == pytest.approx(sigma, rel=1e-9)
    assert cal.demand.revenue_shares(np.array([1.7]))[0] == pytest.approx(w, rel=1e-12)
    assert cal.demand.income == pytest.approx(3.0 / w, rel=1e-12)


def test_ces_calibration_reproduces_shares_and_costs():
    rs = np.array([0.15, 0.25, 0.20, 0.10])
    mk = Market.build(
        P4,
        Shares.total(rs, "revenue"),
        Margins.lerner([0.4, np.nan, np.nan, np.nan]),
        OWNERS4,
        revenue=0.7,
    )
    cal = calibrate_ces(mk)
    np.testing.assert_allclose(cal.demand.revenue_shares(P4), rs, rtol=1e-12)
    assert cal.fitted_margins[0] == pytest.approx(0.4, rel=1e-9)
    assert cal.demand.income == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("margin", [1.0 - 1e-10, 1e-7])
def test_ces_margins_incompatible_with_model_raise(margin):
    mk = Market.build(
        np.array([1.0, 1.0]),
        Shares.total([0.3, 0.3], "revenue"),
        Margins.lerner([margin, np.nan]),
        ["A", "B"],
    )
    with pytest.raises(CalibrationError, match="boundary"):
        calibrate_ces(mk)


# -------------------------------------------------------------------------- linear


def test_linear_calibration_reproduces_quantities_margins_and_diversion():
    d = np.array(
        [[0, 0.30, 0.25, 0.15], [0.20, 0, 0.30, 0.20], [0.25, 0.35, 0, 0.10], [0.30, 0.20, 0.25, 0]]
    )
    q = np.array([15.0, 25.0, 20.0, 10.0]) / P4
    mk = Market.build(
        P4,
        Shares.within(q / q.sum(), "quantity"),
        Margins.lerner([0.4, 0.3, 0.35, 0.45]),
        OWNERS4,
        revenue=float(np.sum(q * P4)),
    )
    cal = calibrate_linear(mk, Diversion.quantity(d))
    np.testing.assert_allclose(cal.demand.quantities(P4), q, rtol=1e-13)
    np.testing.assert_allclose(cal.fitted_margins, [0.4, 0.3, 0.35, 0.45], rtol=1e-12)
    np.testing.assert_allclose(cal.demand.diversion_quantity(P4), d, rtol=1e-12)


def test_linear_requires_every_margin():
    mk = logit_market([0.4, np.nan, np.nan, np.nan])
    mk2 = Market.build(
        P4,
        Shares.within([0.3, 0.3, 0.2, 0.2], "quantity"),
        Margins.lerner([0.4, np.nan, 0.3, 0.3]),
        OWNERS4,
    )
    assert mk.n_products == 4
    with pytest.raises(CalibrationError, match="every product"):
        calibrate_linear(mk2)


def test_linear_rejects_revenue_diversion():
    mk = Market.build(
        P4, Shares.within([0.3, 0.3, 0.2, 0.2], "quantity"), Margins.lerner([0.4] * 4), OWNERS4
    )
    with pytest.raises(ValueError, match="requires quantity diversion"):
        calibrate_linear(mk, Diversion.revenue(np.full((4, 4), 0.2)))


# ------------------------------------------------------------------------- PCAIDS


def test_pcaids_matches_r_elasticities_margins_and_diversion(r_fixtures):
    f = r_fixtures["pcaids_epstein_rubinfeld"]
    p = np.array(f["prices"])
    mk = Market.build(
        p,
        Shares.within(f["revenue_shares"], "revenue"),
        Margins.lerner([np.nan] * 3),
        ["A", "B", "C"],
    )
    cal = calibrate_pcaids(mk, f["own_elasticity_first"], f["market_elasticity"])
    np.testing.assert_allclose(cal.demand.elasticities(p), np.array(f["elasticities"]), rtol=1e-12)
    np.testing.assert_allclose(cal.fitted_margins, f["margins"], rtol=1e-12)
    w = np.array(f["revenue_shares"])
    expected = w[None, :] / (1 - w[:, None])
    q = cal.demand.quantities(p)
    jac = cal.demand.jacobian(p)
    revenue_div = -(jac.T * p[None, :]) / (q + p * np.diag(jac))[:, None]
    np.fill_diagonal(revenue_div, 0.0)
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(revenue_div[off], expected[off], rtol=1e-12)


def test_pcaids_rejects_incompatible_inputs():
    mk = Market.build(
        np.ones(2), Shares.within([0.5, 0.5], "revenue"), Margins.lerner([np.nan] * 2), ["A", "B"]
    )
    with pytest.raises(ValueError, match="below -1"):
        calibrate_pcaids(mk, -0.8)
    with pytest.raises(ValueError, match="requires revenue shares"):
        calibrate_pcaids(
            Market.build(
                np.ones(2),
                Shares.within([0.5, 0.5], "quantity"),
                Margins.lerner([np.nan] * 2),
                ["A", "B"],
            ),
            -2.0,
        )
