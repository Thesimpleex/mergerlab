import numpy as np
import pytest

from mergerlab.units import Diversion, DiversionBasis, Margins, Ownership, Shares


def test_shares_within_must_sum_to_one():
    Shares.within([0.2, 0.3, 0.5], "revenue")
    with pytest.raises(ValueError, match="sum to 1"):
        Shares.within([0.2, 0.3, 0.4], "revenue")


def test_shares_total_infers_outside():
    s = Shares.total([0.2, 0.3], "quantity")
    assert s.outside == pytest.approx(0.5)
    assert s.has_outside
    with pytest.raises(ValueError, match="outside good"):
        Shares.total([0.5, 0.5], "quantity")


def test_basis_conversion_round_trip():
    prices = np.array([2.9, 3.4, 2.2])
    rev = Shares.within([0.2, 0.3, 0.5], "revenue")
    qty = rev.to_quantity(prices)
    assert qty.values.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        qty.values, [0.2 / 2.9, 0.3 / 3.4, 0.5 / 2.2] / np.sum([0.2 / 2.9, 0.3 / 3.4, 0.5 / 2.2])
    )
    np.testing.assert_allclose(qty.to_revenue(prices).values, rev.values, rtol=1e-14)


def test_conversion_with_outside_good_needs_its_price():
    s = Shares.total([0.2, 0.3], "revenue")
    with pytest.raises(ValueError, match="outside good"):
        s.to_quantity([1.0, 2.0])
    converted = s.to_quantity([1.0, 2.0], outside_price=1.5)
    raw = np.array([0.2 / 1.0, 0.3 / 2.0, 0.5 / 1.5])
    np.testing.assert_allclose(converted.values, raw[:2] / raw.sum())
    assert converted.outside == pytest.approx(raw[2] / raw.sum())
    back = converted.to_revenue([1.0, 2.0], outside_price=1.5)
    np.testing.assert_allclose(back.values, s.values, rtol=1e-14)
    assert back.outside == pytest.approx(0.5)
    inside = Shares.within(s.within_values(), "revenue").to_quantity([1.0, 2.0]).with_outside(0.5)
    assert inside.outside == 0.5


def test_require_names_the_fix():
    s = Shares.within([0.5, 0.5], "revenue")
    with pytest.raises(ValueError, match="requires quantity shares"):
        s.require(Shares.within([0.5, 0.5], "quantity").basis, False, "demo")


def test_margins_conventions():
    p = np.array([2.0, 4.0])
    lerner = Margins.lerner([0.25, 0.5])
    absolute = Margins.absolute([0.5, 2.0])
    np.testing.assert_allclose(lerner.as_absolute(p), absolute.as_absolute(p))
    np.testing.assert_allclose(absolute.as_lerner(p), lerner.as_lerner(p))
    with pytest.raises(ValueError, match="absolute margins"):
        Margins.lerner([1.5])
    with pytest.raises(ValueError, match="smaller than prices"):
        Margins.absolute([3.0]).as_lerner([2.0])


def test_diversion_conversions_round_trip():
    prices = np.array([2.9, 3.4, 2.2])
    d = Diversion.quantity([[0, 0.3, 0.4], [0.2, 0, 0.5], [0.25, 0.35, 0]])
    e = np.array([-3.0, -2.75, -2.25])
    value = d.to_value(prices)
    assert value.basis is DiversionBasis.VALUE
    np.testing.assert_allclose(value.matrix[0, 1], 0.3 * 3.4 / 2.9)
    np.testing.assert_allclose(value.to_quantity(prices).matrix, d.matrix, rtol=1e-14)
    rev = d.to_revenue(prices, e)
    np.testing.assert_allclose(rev.to_quantity(prices, e).matrix, d.matrix, rtol=1e-14)
    with pytest.raises(ValueError, match="own-price"):
        rev.to_quantity(prices)


def _er_pcaids(market_elasticity: float):
    """Epstein-Rubinfeld three-firm PCAIDS: prices 2.9 / 3.4 / 2.2, revenue shares 20 / 30 / 50%."""
    from mergerlab.demand import calibrate_pcaids
    from mergerlab.market import Market

    p = np.array([2.9, 3.4, 2.2])
    mk = Market.build(
        p, Shares.within([0.2, 0.3, 0.5], "revenue"), Margins.lerner([np.nan] * 3), list("ABC")
    )
    return p, calibrate_pcaids(mk, -3.0, market_elasticity).demand


@pytest.mark.parametrize("market_elasticity", [-1.0, -1.5, -2.5, -0.7])
def test_share_diversion_converts_to_the_quantity_diversion_of_the_demand_system(
    market_elasticity,
):
    """R's AIDS diversion -B_ji / B_ii (share-slope ratio) against finite-difference truth."""
    p, demand = _er_pcaids(market_elasticity)

    def column_slopes(fun):
        out = np.empty((3, 3))
        for i in range(3):
            h = np.zeros(3)
            h[i] = 1e-6
            out[:, i] = (fun(p + h) - fun(p - h)) / 2e-6
        return out

    dw = column_slopes(demand.revenue_shares)
    dq = column_slopes(demand.quantities)
    r_style = -dw.T / np.diag(dw)[:, None]
    truth = -dq.T / np.diag(dq)[:, None]
    e = np.diag(demand.elasticities(p))
    w = demand.revenue_shares(p)
    converted = Diversion.share(r_style).to_quantity(
        p, e, shares=w, market_elasticity=market_elasticity
    )
    mask = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(converted.matrix[mask], truth[mask], rtol=1e-6)
    back = converted.to_share(p, e, w, market_elasticity)
    np.testing.assert_allclose(back.matrix, np.where(mask, r_style, 0.0), rtol=1e-6)


def test_share_and_revenue_diversion_agree_only_at_market_elasticity_minus_one():
    for eps_m, agree in ((-1.0, True), (-1.5, False)):
        p, demand = _er_pcaids(eps_m)
        e = np.diag(demand.elasticities(p))
        q = Diversion.from_demand(demand.diversion_quantity(p))
        share = q.to_share(p, e, demand.revenue_shares(p), eps_m).matrix
        revenue = q.to_revenue(p, e).matrix
        assert np.allclose(share, revenue) is agree
    p, demand = _er_pcaids(-1.5)
    assert Diversion.from_demand(demand.diversion_quantity(p)).matrix[0, 1] == pytest.approx(
        0.1599, abs=5e-4
    )


def test_share_diversion_conversion_needs_shares_and_market_elasticity():
    share = Diversion.share([[0, 0.375], [0.2857, 0]])
    with pytest.raises(ValueError, match="market_elasticity"):
        share.to_quantity([2.9, 3.4], [-3.0, -2.75])
    with pytest.raises(ValueError, match="share"):
        share.require_quantity("CMCR")
    with pytest.raises(ValueError, match="compatible"):
        share.to_quantity([2.9, 3.4], [-1.05, -1.05], shares=[0.5, 0.5], market_elasticity=-4.0)


def test_demand_derived_quantity_diversion_may_exceed_one_but_user_input_may_not():
    row_sums = Diversion.from_demand([[0, 0.7, 0.6], [0.1, 0, 0.1], [0.1, 0.1, 0]]).matrix.sum(1)
    assert row_sums[0] == pytest.approx(1.3)
    with pytest.raises(ValueError, match="cannot sum to more than 1"):
        Diversion.quantity([[0, 0.7, 0.6], [0.1, 0, 0.1], [0.1, 0.1, 0]])


def test_revenue_diversion_refused_where_quantity_needed():
    rev = Diversion.revenue([[0, 0.375], [0.2857, 0]])
    with pytest.raises(ValueError, match="requires quantity diversion"):
        rev.require_quantity("CMCR")


def test_quantity_diversion_rows_cannot_exceed_one():
    with pytest.raises(ValueError, match="cannot sum to more than 1"):
        Diversion.quantity([[0, 0.7, 0.6], [0.1, 0, 0.1], [0.1, 0.1, 0]])


def test_ownership_merge_and_validation():
    own = Ownership.from_owners(["A", "B", "C", "B"])
    post = own.merged(["A", "B"])
    np.testing.assert_array_equal(
        post.matrix,
        [[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 0, 1]],
    )
    np.testing.assert_array_equal(own.merging_products(post), [True, True, False, True])
    with pytest.raises(ValueError, match="diagonal"):
        Ownership(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unknown owner"):
        own.merged(["A", "Z"])


def test_quantity_shares_within_divide_revenue_shares_by_price():
    from mergerlab.market import Market

    prices = np.array([1.7, 0.6, 1.1])
    w = np.array([0.5, 0.2, 0.3])
    mk = Market.build(prices, Shares.within(w, "revenue"), Margins.lerner([0.3] * 3), list("ABC"))
    expected = (w / prices) / np.sum(w / prices)
    np.testing.assert_allclose(mk.quantity_shares_within(), expected, rtol=1e-14)
    np.testing.assert_allclose(mk.revenue_shares_within(), w, rtol=1e-14)
    qs = Shares.within(expected, "quantity")
    mk2 = Market.build(prices, qs, Margins.lerner([0.3] * 3), list("ABC"))
    np.testing.assert_allclose(mk2.revenue_shares_within(), w, rtol=1e-12)
