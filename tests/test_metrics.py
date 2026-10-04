import numpy as np
import pytest
from scipy import optimize

from helpers import random_logit_market
from mergerlab.demand import (
    Logit,
    NestedLogit,
    calibrate_ces,
    calibrate_linear,
    calibrate_logit,
    calibrate_nested_logit,
    calibrate_pcaids,
)
from mergerlab.errors import CalibrationError
from mergerlab.market import Market
from mergerlab.metrics import (
    cmcr_from_demand,
    cmcr_from_diversion,
    cmcr_two_product,
    compensating_variation,
    cost_pressure,
    critical_efficiency,
    critical_loss,
    diversion_from_demand,
    first_order_price_effects,
    guppi,
    guppi_matrix,
    hypothetical_monopolist_test,
    merger_pass_through,
    pricing_pressure,
    producer_surplus,
    synergy_amounts,
    upp,
)
from mergerlab.supply import scaled_residual, solve_bertrand
from mergerlab.units import Diversion, DiversionBasis, Margins, Ownership, Shares


def miller_example():
    """Three firms, margin 0.5, share 0.3 each, outside good 0.1 (Miller et al., section 2.2)."""
    p = np.ones(3)
    mk = Market.build(
        p, Shares.total([0.3, 0.3, 0.3], "quantity"), Margins.lerner([0.5] * 3), list("ABC")
    )
    cal = calibrate_logit(mk)
    return mk, cal, mk.ownership, mk.ownership.merged(["A", "B"])


# -------------------------------------------------------------------- GUPPI / UPP


def test_guppi_two_product_closed_form():
    p = np.array([2.0, 3.0])
    d = Diversion.quantity([[0, 0.3], [0.2, 0]])
    m = Margins.lerner([0.4, 0.5])
    g = guppi_matrix(p, m, d)
    assert g[0, 1] == pytest.approx(0.3 * 0.5 * 3.0 / 2.0)
    assert g[1, 0] == pytest.approx(0.2 * 0.4 * 2.0 / 3.0)
    own = Ownership.from_owners(["A", "B"])
    np.testing.assert_allclose(guppi(p, m, d, own, own.merged(["A", "B"])), [g[0, 1], g[1, 0]])


def test_guppi_rejects_revenue_diversion_and_accepts_converted_one():
    p = np.array([2.9, 3.4])
    rev = Diversion.revenue([[0, 0.375], [0.2857, 0]])
    m = Margins.lerner([0.33, 0.36])
    with pytest.raises(ValueError, match="requires quantity diversion"):
        guppi_matrix(p, m, rev)
    q = rev.to_quantity(p, [-3.0, -2.75])
    assert guppi_matrix(p, m, q)[0, 1] == pytest.approx(q.matrix[0, 1] * 0.36 * 3.4 / 2.9)


def test_guppi_matches_model_pricing_pressure_for_single_product_firms():
    mk, cal, pre, post = miller_example()
    div = Diversion.quantity(cal.demand.diversion_quantity(mk.prices))
    g = guppi(mk.prices, Margins.lerner(cal.fitted_margins), div, pre, post)
    pressure = pricing_pressure(cal.demand, mk.prices, cal.costs, pre, post)
    np.testing.assert_allclose(g * mk.prices, pressure, rtol=1e-12)
    assert g[0] == pytest.approx(0.3 / 0.7 * 0.5, rel=1e-12)  # 0.214, Miller et al.


def test_upp_and_critical_efficiency():
    p = np.array([2.0, 3.0])
    m = Margins.lerner([0.4, 0.5])
    g = np.array([0.12, 0.08])
    e_star = critical_efficiency(g, m, p)
    np.testing.assert_allclose(e_star, g / (1 - np.array([0.4, 0.5])))
    np.testing.assert_allclose(upp(g, m, p, e_star), 0.0, atol=1e-15)
    assert np.all(upp(g, m, p, 0.0) == g)


# ------------------------------------------------------------------- CMCR (Werden)


def test_cmcr_general_matrix_form_equals_two_product_closed_form():
    p = np.array([2.9, 3.4])
    d = np.array([[0, 0.21], [0.35, 0]])
    m = [0.33, 0.45]
    own = Ownership.from_owners(["A", "B"])
    res = cmcr_from_diversion(
        p, Margins.lerner(m), Diversion.quantity(d), own, own.merged(["A", "B"])
    )
    c1 = cmcr_two_product(m[0], m[1], d[0, 1], d[1, 0], p[1] / p[0])
    c2 = cmcr_two_product(m[1], m[0], d[1, 0], d[0, 1], p[0] / p[1])
    np.testing.assert_allclose(res.relative, [c1, c2], rtol=1e-12)


def test_convention_trap_revenue_diversion_in_quantity_formula():
    """R's AIDS/PCAIDS revenue diversion (0.375) in Werden's quantity formula: 32.9% not 16.7%."""
    p = np.array([2.9, 3.4])
    m1, m2 = 1 / 3, 1 / 2.75
    wrong = cmcr_two_product(m1, m2, 0.375, 0.2857142857, p[1] / p[0])
    assert wrong == pytest.approx(0.3286, abs=5e-4)
    rev = Diversion.revenue([[0, 0.375], [0.2857142857142857, 0]])
    quantity = rev.to_quantity(p, [-3.0, -2.75])
    right = cmcr_two_product(m1, m2, quantity.matrix[0, 1], quantity.matrix[1, 0], p[1] / p[0])
    assert right == pytest.approx(1 / 6, rel=1e-10)
    own = Ownership.from_owners(["A", "B"])
    res = cmcr_from_diversion(p, Margins.lerner([m1, m2]), quantity, own, own.merged(["A", "B"]))
    assert res.relative[0] == pytest.approx(right, rel=1e-12)


def test_cmcr_two_product_validation():
    with pytest.raises(ValueError, match="Lerner margin"):
        cmcr_two_product(1.2, 0.3, 0.2, 0.2)
    with pytest.raises(ValueError, match="D12 D21 < 1"):
        cmcr_two_product(0.3, 0.3, 1.0, 1.0)


def all_calibrations():
    """One calibration of each demand system on a common four-product market."""
    p = np.array([1.0, 1.3, 0.9, 1.1])
    q_sh = [0.15, 0.25, 0.20, 0.10]
    out = []
    mk = Market.build(
        p,
        Shares.total(q_sh, "quantity"),
        Margins.lerner([0.4, np.nan, np.nan, np.nan]),
        list("ABCD"),
    )
    out.append(calibrate_logit(mk))
    out.append(calibrate_nested_logit(mk, [0, 0, 1, 1], 0.5))
    mk_rev = Market.build(
        p,
        Shares.total(np.array(q_sh) * p / np.sum(np.array(q_sh) * p) * 0.7, "revenue"),
        Margins.lerner([0.4, np.nan, np.nan, np.nan]),
        list("ABCD"),
        revenue=1.0,
    )
    out.append(calibrate_ces(mk_rev))
    mk_lin = Market.build(
        p,
        Shares.within([0.3, 0.3, 0.2, 0.2], "quantity"),
        Margins.lerner([0.4, 0.3, 0.35, 0.45]),
        list("ABCD"),
    )
    out.append(calibrate_linear(mk_lin))
    mk_pc = Market.build(
        p,
        Shares.within([0.3, 0.3, 0.2, 0.2], "revenue"),
        Margins.lerner([np.nan] * 4),
        list("ABCD"),
    )
    out.append(calibrate_pcaids(mk_pc, -2.5, -1.0))
    return out


@pytest.mark.parametrize("cal", all_calibrations(), ids=lambda c: c.name)
def test_cmcr_adjusted_costs_reproduce_pre_merger_prices_exactly(cal):
    mk = cal.market
    post = mk.ownership.merged(["A", "C"])
    cm = cmcr_from_demand(cal.demand, mk.prices, cal.costs, post)
    assert np.all(cm.relative[[0, 2]] > 0)
    assert cm.relative[1] == pytest.approx(0, abs=1e-12)
    eq = solve_bertrand(cal.demand, cm.costs_neutral, post, mk.prices)
    np.testing.assert_allclose(eq.prices, mk.prices, rtol=1e-9)


def test_cmcr_from_diversion_equals_cmcr_from_demand():
    mk, cal, pre, post = miller_example()
    div = Diversion.quantity(cal.demand.diversion_quantity(mk.prices))
    a = cmcr_from_diversion(mk.prices, Margins.lerner(cal.fitted_margins), div, pre, post)
    b = cmcr_from_demand(cal.demand, mk.prices, cal.costs, post)
    np.testing.assert_allclose(a.relative, b.relative, rtol=1e-12)
    np.testing.assert_allclose(a.level, b.level, rtol=1e-12)


# ------------------------------------------------------- first-order approximation


def test_miller_et_al_illustration_values():
    """Section 2.2 of Miller, Remer, Ryan and Sheu: UPP 0.214, FOA 0.204, simulation 0.190."""
    mk, cal, pre, post = miller_example()
    pressure = pricing_pressure(cal.demand, mk.prices, cal.costs, pre, post)
    foa = first_order_price_effects(cal.demand, mk.prices, cal.costs, pre, post)
    eq = solve_bertrand(cal.demand, cal.costs, post, mk.prices)
    sim = eq.prices - mk.prices
    np.testing.assert_allclose(pressure, [0.214, 0.214, 0.0], atol=5e-4)
    np.testing.assert_allclose(foa, [0.204, 0.204, 0.052], atol=5e-4)
    np.testing.assert_allclose(sim, [0.190, 0.190, 0.052], atol=5e-4)
    rho = merger_pass_through(cal.demand, mk.prices, cal.costs, pre, post)
    np.testing.assert_allclose(
        rho, [[0.771, 0.180, 0.297], [0.180, 0.771, 0.297], [0.122, 0.122, 0.776]], atol=5e-4
    )
    np.testing.assert_allclose(foa, rho @ pressure, rtol=1e-12)


def test_first_order_error_is_second_order_in_merger_strength():
    """Partial ownership change theta: |FOA - simulation| shrinks like theta^2."""
    mk, cal, pre, post = miller_example()

    def error(theta: float) -> float:
        partial = Ownership(pre.matrix + theta * (post.matrix - pre.matrix))
        foa = first_order_price_effects(cal.demand, mk.prices, cal.costs, pre, partial)
        sim = solve_bertrand(cal.demand, cal.costs, partial, mk.prices).prices - mk.prices
        return float(np.max(np.abs(foa - sim)))

    e1, e2, e3 = error(0.2), error(0.1), error(0.05)
    assert e2 / e1 == pytest.approx(0.25, rel=0.15)
    assert e3 / e2 == pytest.approx(0.25, rel=0.15)


def test_log_price_form_of_the_first_order_approximation():
    """Koh's log-price pass-through: also second-order accurate, and equal to the level form
    at leading order (the two differ only through the O(g^2) term)."""
    mk, cal, pre, post = miller_example()

    def errors(theta: float) -> tuple[float, float, float]:
        partial = Ownership(pre.matrix + theta * (post.matrix - pre.matrix))
        args = (cal.demand, mk.prices, cal.costs, pre, partial)
        level = first_order_price_effects(*args)
        log = first_order_price_effects(*args, log_prices=True)
        sim = solve_bertrand(cal.demand, cal.costs, partial, mk.prices).prices - mk.prices
        return (
            float(np.max(np.abs(log - sim))),
            float(np.max(np.abs(log - level))),
            float(np.max(np.abs(sim))),
        )

    e1, d1, _ = errors(0.1)
    e2, d2, _ = errors(0.05)
    assert e2 / e1 == pytest.approx(0.25, rel=0.2)
    assert d2 / d1 == pytest.approx(0.25, rel=0.2)
    m = merger_pass_through(cal.demand, mk.prices, cal.costs, pre, post, log_prices=True)
    assert m.shape == (3, 3)


def _unequal_price_ces():
    p = np.array([1.6, 1.3, 1.0])
    mk = Market.build(
        p,
        Shares.total([0.30, 0.20, 0.35], "revenue"),
        Margins.lerner([0.4, np.nan, np.nan]),
        list("ABC"),
    )
    cal = calibrate_ces(mk)
    return mk, cal, mk.ownership, mk.ownership.merged(["A", "B"])


def test_cost_pressure_single_product_firms_is_cost_change_less_diverted_cost_change():
    """Row j of the efficiency term is dc_j - D_jk dc_k for merging single-product firms."""
    mk, cal, pre, post = _unequal_price_ces()
    dc = np.array([-0.08, -0.03, 0.0])
    t = cost_pressure(cal.demand, mk.prices, pre, post, dc)
    d = cal.demand.diversion_quantity(mk.prices)
    np.testing.assert_allclose(t[0], dc[0] - d[0, 1] * dc[1], rtol=1e-12)
    np.testing.assert_allclose(t[1], dc[1] - d[1, 0] * dc[0], rtol=1e-12)
    assert t[2] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("log_prices", [False, True])
@pytest.mark.parametrize("market", ["logit", "ces_unequal_prices"])
def test_first_order_cost_effect_converges_to_simulated_cost_effect(market, log_prices):
    """(FOA(e) - FOA(0)) / (sim(e) - sim(0)) tends to one as the merger weakens (theta -> 0).

    The approximation is taken at the pre-merger prices, so for a full merger it differs from
    the simulated effect by terms of the order of the merger itself. Adding the cost change
    instead of its pass-through to pricing pressure leaves a ratio far from one (1.75 in the
    Miller et al. example) that does not shrink with theta.
    """
    mk, cal, pre, post = miller_example() if market == "logit" else _unequal_price_ces()
    p0 = mk.prices
    merging = np.array([True, True, False])
    e = 0.001

    def ratio(theta: float) -> float:
        partial = Ownership(pre.matrix + theta * (post.matrix - pre.matrix))
        args = (cal.demand, p0, cal.costs, pre, partial)
        foa = first_order_price_effects(
            *args, cost_change=-e * merging * cal.costs, log_prices=log_prices
        )
        foa0 = first_order_price_effects(*args, log_prices=log_prices)
        sim = solve_bertrand(cal.demand, cal.costs * (1.0 - e * merging), partial, p0).prices
        sim0 = solve_bertrand(cal.demand, cal.costs, partial, p0).prices
        return float(((foa - foa0) / (sim - sim0))[0])

    errors = [abs(ratio(theta) - 1.0) for theta in (0.2, 0.1, 0.05)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.015
    if not log_prices:
        assert abs(ratio(1.0) - 1.0) < 0.05


def test_cost_savings_enter_the_first_order_approximation_linearly():
    mk, cal, pre, post = miller_example()
    saving = np.array([-0.05, -0.05, 0.0])
    foa = first_order_price_effects(cal.demand, mk.prices, cal.costs, pre, post, saving)
    base = first_order_price_effects(cal.demand, mk.prices, cal.costs, pre, post)
    rho = merger_pass_through(cal.demand, mk.prices, cal.costs, pre, post)
    t = cost_pressure(cal.demand, mk.prices, pre, post, saving)
    np.testing.assert_allclose(foa - base, rho @ t, rtol=1e-12)
    assert foa[0] < base[0]


def test_log_price_pass_through_matches_finite_differences_at_unequal_prices():
    """M = -(d (h / p) / d ln p)^{-1}, h the post-merger first-order condition in markup form."""
    p = np.array([1.7, 0.6, 1.1])
    demand = Logit(np.array([2.2, 1.0, 1.6]), 1.4, 1.0)
    costs = np.array([0.9, 0.3, 0.6])
    pre = Ownership.from_owners(["A", "B", "C"])
    post = pre.merged(["A", "B"])

    def h_over_p(x):
        jac = demand.jacobian(x)
        f_post = demand.quantities(x) + (post.matrix * jac.T) @ (x - costs)
        return -np.linalg.solve(pre.matrix * jac.T, f_post) / x

    step = 1e-6
    d = np.empty((3, 3))
    for k in range(3):
        up, down = np.log(p), np.log(p)
        up[k] += step
        down[k] -= step
        d[:, k] = (h_over_p(np.exp(up)) - h_over_p(np.exp(down))) / (2 * step)
    expected = -np.linalg.inv(d)
    got = merger_pass_through(demand, p, costs, pre, post, log_prices=True)
    np.testing.assert_allclose(got, expected, rtol=1e-6)


# --------------------------------------------------------------- invariants


def test_unchanged_ownership_gives_zero_effects():
    rng = np.random.default_rng(5)
    done = 0
    while done < 10:
        mk = random_logit_market(rng, 5, firms=4)
        try:
            cal = calibrate_logit(mk)
        except CalibrationError:
            continue
        eq = solve_bertrand(cal.demand, cal.costs, mk.ownership, mk.prices * 1.2)
        np.testing.assert_allclose(eq.prices, mk.prices, rtol=1e-10)
        assert np.all(
            first_order_price_effects(cal.demand, mk.prices, cal.costs, mk.ownership, mk.ownership)
            == 0
        )
        done += 1


def test_divesting_every_overlap_restores_pre_merger_prices():
    mk, cal, _, post = miller_example()
    merged = solve_bertrand(cal.demand, cal.costs, post, mk.prices)
    assert merged.prices[0] > mk.prices[0] + 0.05
    restored = solve_bertrand(
        cal.demand, cal.costs, Ownership.from_owners(["A", "B", "C"]), merged.prices
    )
    np.testing.assert_allclose(restored.prices, mk.prices, rtol=1e-10)


def test_symmetric_merger_gives_symmetric_effects_and_higher_prices():
    p = np.ones(5)
    mk = Market.build(
        p,
        Shares.total([0.15] * 5, "quantity"),
        Margins.lerner([0.45, np.nan, np.nan, np.nan, np.nan]),
        list("ABCDE"),
    )
    for model in (
        calibrate_logit(mk),
        calibrate_nested_logit(mk, [0, 0, 0, 1, 1], 0.4),
    ):
        post = mk.ownership.merged(["A", "B"])
        eq = solve_bertrand(model.demand, model.costs, post, p)
        if model.name == "logit":
            assert eq.prices[0] == pytest.approx(eq.prices[1], rel=1e-10)
            assert eq.prices[2] == pytest.approx(eq.prices[3], rel=1e-10)
        assert eq.prices[:2].min() > eq.prices[2:].max()


def test_nested_logit_stronger_nest_correlation_raises_within_nest_price_effect():
    mk = Market.build(
        np.ones(4),
        Shares.total([0.2] * 4, "quantity"),
        Margins.lerner([0.5, np.nan, np.nan, np.nan]),
        list("ABCD"),
    )
    effects = []
    for sigma in (0.0, 0.4, 0.8):
        cal = calibrate_nested_logit(mk, [0, 0, 1, 1], sigma)
        post = mk.ownership.merged(["A", "B"])
        eq = solve_bertrand(cal.demand, cal.costs, post, mk.prices)
        effects.append(eq.prices[0] - 1.0)
    assert effects[0] < effects[1] < effects[2]


# ------------------------------------------------------------ hypothetical monopolist


def test_critical_loss_formula_and_equivalence_with_profit_test():
    assert critical_loss(0.05, 0.4) == pytest.approx(0.05 / 0.45)
    with pytest.raises(ValueError, match="lerner_margin"):
        critical_loss(0.05, 1.2)
    demand = Logit(np.array([2.0, 1.5]), 1.8, 1.0)
    costs = np.array([0.5, 0.6])
    p = np.array([1.3, 1.4])
    res = hypothetical_monopolist_test(demand, p, costs, [0, 1], ssnip=0.05)
    margin = float(((p - costs) * demand.quantities(p)).sum() / (p * demand.quantities(p)).sum())
    assert res.critical_loss == pytest.approx(0.05 / (0.05 + margin))
    # unequal margins: shortcut and exact profit test can differ only marginally; with equal
    # margins they are equivalent
    equal = hypothetical_monopolist_test(demand, p, np.array([0.7, 0.7]), [0, 1], ssnip=0.05)
    assert equal.ssnip_profitable == (equal.actual_loss < equal.critical_loss)


def test_hypothetical_monopolist_single_product_matches_scalar_optimisation():
    delta = np.array([2.0, 1.2, 1.0])
    demand = Logit(delta, 1.6, 1.0)
    costs = np.array([0.5, 0.6, 0.55])
    p = np.array([1.2, 1.1, 1.0])

    def neg_profit(x: float) -> float:
        pp = p.copy()
        pp[0] = x
        return -(x - costs[0]) * demand.quantities(pp)[0]

    best = optimize.minimize_scalar(
        neg_profit, bounds=(0.5, 5), method="bounded", options={"xatol": 1e-12}
    )
    res = hypothetical_monopolist_test(demand, p, costs, [0], ssnip=0.05)
    assert res.price_increase[0] == pytest.approx(best.x / p[0] - 1, rel=1e-6)
    assert res.passes == (best.x / p[0] - 1 >= 0.05)


def test_hypothetical_monopolist_passes_if_any_product_reaches_the_ssnip():
    """Candidate {1, 2}: product 1's optimal price falls, product 2's rises by more than 5%."""
    demand = Logit(np.array([2.0, 1.5, 1.0]), 1.8, 1.0)
    costs = np.array([0.5, 0.6, 0.55])
    p = np.array([1.3, 1.4, 1.2])
    res = hypothetical_monopolist_test(demand, p, costs, [1, 2], ssnip=0.05)
    assert res.price_increase.min() < 0.05 <= res.price_increase.max()
    assert res.passes
    assert res.max_increase == pytest.approx(res.price_increase.max())
    below = hypothetical_monopolist_test(demand, p, costs, [0], ssnip=0.05)
    assert below.price_increase.max() < 0.05 and not below.passes


def test_hypothetical_monopolist_loss_and_profit_follow_hand_computation():
    demand = Logit(np.array([2.0, 1.5, 1.0]), 1.8, 1.0)
    costs = np.array([0.5, 0.6, 0.55])
    p = np.array([1.3, 1.4, 1.2])
    idx = [1, 2]
    res = hypothetical_monopolist_test(demand, p, costs, idx, ssnip=0.05)
    p_s = p.copy()
    p_s[idx] *= 1.05
    q0, q1 = demand.quantities(p), demand.quantities(p_s)
    revenue = p[idx] * q0[idx]
    loss = np.sum(revenue * (1 - q1[idx] / q0[idx])) / revenue.sum()
    margin = np.sum((p - costs)[idx] * q0[idx]) / revenue.sum()
    assert res.actual_loss == pytest.approx(loss, rel=1e-12)
    assert res.critical_loss == pytest.approx(0.05 / (0.05 + margin), rel=1e-12)
    profit0 = np.sum(((p - costs) * q0)[idx])
    profit1 = np.sum(((p_s - costs) * q1)[idx])
    assert res.ssnip_profitable == bool(profit1 > profit0)
    assert not res.ssnip_profitable and res.actual_loss > res.critical_loss


def test_hypothetical_monopolist_larger_candidate_market_raises_prices_more():
    demand = NestedLogit(np.array([2.0, 1.8, 1.5, 1.4]), 1.5, 0.5, [0, 0, 0, 0], 1.0)
    own = Ownership.from_owners(list("ABCD"))
    costs = np.array([0.5, 0.5, 0.5, 0.5])
    p = solve_bertrand(demand, costs, own, costs + 1).prices
    one = hypothetical_monopolist_test(demand, p, costs, [0])
    two = hypothetical_monopolist_test(demand, p, costs, [0, 1])
    allp = hypothetical_monopolist_test(demand, p, costs, [0, 1, 2, 3])
    assert one.max_increase < two.max_increase < allp.max_increase
    respond = hypothetical_monopolist_test(
        demand, p, costs, [0, 1], rivals="respond", ownership=own
    )
    assert respond.max_increase > two.max_increase  # strategic complements amplify


# ------------------------------------------------------------------------ welfare


def test_consumer_harm_and_producer_surplus_signs():
    mk, cal, _, post = miller_example()
    eq = solve_bertrand(cal.demand, cal.costs, post, mk.prices)
    assert compensating_variation(cal.demand, mk.prices, eq.prices) > 0
    ps0 = producer_surplus(cal.demand, mk.prices, cal.costs)
    ps1 = producer_surplus(cal.demand, eq.prices, cal.costs)
    assert ps1[:2].sum() > ps0[:2].sum()
    assert scaled_residual(cal.demand, eq.prices, cal.costs, post) < 1e-10


# --------------------------------------------------------- diversion and synergies


def test_diversion_bases_from_demand_round_trip(r_fixtures):
    f = r_fixtures["pcaids_epstein_rubinfeld"]
    p = np.array(f["prices"])
    mk = Market.build(
        p, Shares.within(f["revenue_shares"], "revenue"), Margins.lerner([np.nan] * 3), list("ABC")
    )
    cal = calibrate_pcaids(mk, f["own_elasticity_first"], f["market_elasticity"])
    rev = diversion_from_demand(cal.demand, p, DiversionBasis.REVENUE)
    off = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(rev.matrix[off], np.array(f["revenue_diversion"])[off], rtol=1e-10)
    own = np.diag(cal.demand.elasticities(p))
    quantity = diversion_from_demand(cal.demand, p)
    np.testing.assert_allclose(rev.to_quantity(p, own).matrix, quantity.matrix, rtol=1e-12)
    value = diversion_from_demand(cal.demand, p, DiversionBasis.VALUE)
    np.testing.assert_allclose(value.matrix[0, 1], quantity.matrix[0, 1] * p[1] / p[0])
    assert quantity.matrix[0, 1] == pytest.approx(0.21324, abs=5e-5)  # not 0.375


def test_synergy_amounts():
    mk, cal, _, post = miller_example()
    cm = cmcr_from_demand(cal.demand, mk.prices, cal.costs, post)
    q = cal.demand.quantities(mk.prices) * 100.0
    per_unit, total = synergy_amounts(cm, q)
    np.testing.assert_allclose(per_unit, cm.level)
    assert total == pytest.approx(float(np.sum(cm.level[:2] * q[:2])), rel=1e-14)
    with pytest.raises(ValueError, match="one entry per product"):
        synergy_amounts(cm, [1.0])
