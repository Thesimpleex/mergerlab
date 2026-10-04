import numpy as np
import pytest

from mergerlab.market import Market
from mergerlab.simulate import ModelSpec, calibrate, simulate_merger, simulate_one
from mergerlab.uncertainty import Priors, run_monte_carlo, weighted_quantile
from mergerlab.units import Margins, Shares

P3 = np.ones(3)


def heinz_like(heinz_margin=0.25):
    return Market.build(
        P3,
        Shares.total([0.65, 0.174, 0.154], "revenue"),
        Margins.lerner([np.nan, heinz_margin, np.nan]),
        ["Gerber", "Heinz", "Beech-Nut"],
        revenue=865.0,
        outside_price=1.0,
    )


SPECS = [
    ModelSpec("logit"),
    ModelSpec("nested_logit", nests=(0, 1, 1), sigma=0.4),
    ModelSpec("ces"),
    ModelSpec("linear", fill_margins_from="logit"),
    ModelSpec("pcaids", known_index=1),
]


def test_simulate_merger_runs_all_forms_and_reports_screens():
    mk = heinz_like()
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    res = simulate_merger(mk, post, SPECS)
    assert [o.status for o in res.outcomes] == ["ok"] * 5
    assert res.parties == (1, 2)
    assert res.concentration.delta_hhi == pytest.approx(2e4 * 0.174 * 0.154)
    assert {s.ruleset for s in res.screens} == {"us2023", "us2010", "eu2004"}
    for o in res.outcomes:
        assert o.residual is not None and o.residual < 1e-10
        assert np.all(o.price_change[1:] > 0)
        assert o.cmcr is not None and o.cmcr.relative[0] == pytest.approx(0, abs=1e-12)
        assert o.consumer_harm is not None and o.consumer_harm > 0
        assert o.required_synergy_total is not None and o.required_synergy_total > 0
    lo, hi = res.price_effect_range()
    assert 0 < lo < hi


def test_quantities_scale_with_revenue_not_prices():
    a = simulate_merger(
        heinz_like(), heinz_like().ownership.merged(["Heinz", "Beech-Nut"]), SPECS[:1]
    )
    mk = heinz_like()
    mk2 = Market.build(
        P3,
        mk.shares,
        mk.margins,
        ["Gerber", "Heinz", "Beech-Nut"],
        revenue=1730.0,
        outside_price=1.0,
    )
    b = simulate_merger(mk2, mk2.ownership.merged(["Heinz", "Beech-Nut"]), SPECS[:1])
    np.testing.assert_allclose(a.outcomes[0].price_change, b.outcomes[0].price_change, rtol=1e-12)
    assert b.outcomes[0].consumer_harm == pytest.approx(2 * a.outcomes[0].consumer_harm, rel=1e-12)
    assert b.outcomes[0].required_synergy_total == pytest.approx(
        2 * a.outcomes[0].required_synergy_total, rel=1e-12
    )


def test_efficiency_equal_to_cmcr_neutralises_the_merger():
    mk = heinz_like()
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    for spec in SPECS:
        base = simulate_one(spec, mk, post)
        eff = base.cmcr.relative
        out = simulate_one(spec, mk, post, efficiency=eff)
        np.testing.assert_allclose(out.price_change, 0.0, atol=1e-9)


def test_failures_are_returned_not_raised_or_dropped():
    mk = heinz_like(heinz_margin=0.45)
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    out = simulate_one(ModelSpec("logit"), mk, post)
    assert out.status == "calibration_failed"
    assert "Lerner margins outside" in out.message
    ok = simulate_one(ModelSpec("pcaids", known_index=1), mk, post)
    assert ok.status == "ok"


def test_calibrate_explains_missing_outside_good():
    mk = Market.build(
        P3,
        Shares.within([0.5, 0.3, 0.2], "revenue"),
        Margins.lerner([0.3, np.nan, np.nan]),
        list("ABC"),
    )
    with pytest.raises(ValueError, match="needs an outside good"):
        calibrate(ModelSpec("logit"), mk)
    alm = Market.build(
        P3,
        Shares.within([0.5, 0.3, 0.2], "quantity"),
        Margins.lerner([1 / (3 * 0.65), 1 / (3 * 0.79), np.nan]),
        list("ABC"),
    )
    cal = calibrate(ModelSpec("logit_alm"), alm)
    assert cal.parameters["outside_share"] == pytest.approx(0.3, rel=1e-8)


def test_linear_margin_fill_is_explicit_and_noted():
    mk = heinz_like()
    with pytest.raises(Exception, match="every product"):
        calibrate(ModelSpec("linear"), mk)
    cal = calibrate(ModelSpec("linear", fill_margins_from="logit"), mk)
    assert "taken from the logit calibration" in cal.notes[0]
    logit = calibrate(ModelSpec("logit"), mk)
    np.testing.assert_allclose(cal.fitted_margins, logit.fitted_margins, rtol=1e-10)


def test_no_change_in_ownership_is_rejected():
    mk = heinz_like()
    with pytest.raises(ValueError, match="does not differ"):
        simulate_merger(mk, mk.ownership, SPECS[:1])


def test_model_spec_validation():
    with pytest.raises(ValueError, match="unknown demand model"):
        ModelSpec("translog")
    with pytest.raises(ValueError, match="needs nests and sigma"):
        ModelSpec("nested_logit")


# --------------------------------------------------------------------- Monte Carlo


def test_monte_carlo_is_reproducible_and_accounts_for_every_draw():
    mk = heinz_like()
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    pri = Priors(0.10, (0.1, 0.7), (-1.5, -0.5), 30.0)
    a = run_monte_carlo(mk, post, SPECS, pri, draws=120, seed=7)
    b = run_monte_carlo(mk, post, SPECS, pri, draws=120, seed=7)
    for k in a.party_change:
        np.testing.assert_array_equal(a.party_change[k], b.party_change[k])
        assert sum(a.counts(k).values()) == 120
        assert np.sum(np.isfinite(a.party_change[k])) == a.counts(k)["ok"]
    assert a.counts("nested_logit")["calibration_failed"] > 0
    assert 0 < a.failure_share() < 1
    lo, mid, hi = a.bands()
    assert lo < mid < hi
    c = run_monte_carlo(mk, post, SPECS, pri, draws=120, seed=8)
    assert not np.allclose(
        np.nan_to_num(a.party_change["logit"]), np.nan_to_num(c.party_change["logit"])
    )


def test_monte_carlo_zero_width_priors_reproduce_point_estimates():
    mk = heinz_like()
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    res = simulate_merger(mk, post, SPECS[:3])
    mc = run_monte_carlo(mk, post, SPECS[:3], Priors(0.0), draws=5, seed=1)
    for o in res.outcomes:
        assert np.allclose(mc.party_change[o.spec.kind], res.party_price_change(o), rtol=1e-12)


def test_monte_carlo_counts_total_calibration_failure():
    mk = heinz_like(heinz_margin=0.45)
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    mc = run_monte_carlo(mk, post, SPECS[:1], Priors(0.0), draws=10, seed=1)
    assert mc.counts("logit") == {"ok": 0, "calibration_failed": 10, "solver_failed": 0}
    assert mc.failure_share() == 1.0
    assert np.all(np.isnan(mc.bands()))


def test_weighted_quantile_equals_hazen_quantile_for_equal_weights():
    rng = np.random.default_rng(0)
    v = rng.normal(size=401)
    q = [0.01, 0.05, 0.5, 0.95, 0.99]
    np.testing.assert_allclose(
        weighted_quantile(v, np.ones_like(v), q), np.quantile(v, q, method="hazen"), rtol=1e-12
    )


def test_weighted_quantile_known_unequal_weight_example():
    """Cumulative midpoints (1/2, 3/2 + 1/2, ...)/4 = 0.125, 0.5, 0.875 for weights 1, 2, 1."""
    v, w = np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.0])
    np.testing.assert_allclose(
        weighted_quantile(v, w, [0.125, 0.3125, 0.5, 0.6875, 0.875]), [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    shuffled = weighted_quantile(v[::-1], w[::-1], [0.3125])
    np.testing.assert_allclose(shuffled, [0.5])
    w2 = np.concatenate([np.ones(10), np.full(10, 1e-9)])
    x = np.concatenate([np.zeros(10), np.full(10, 100.0)])
    assert weighted_quantile(x, w2, [0.5])[0] == pytest.approx(0.0, abs=1e-6)


def test_priors_validation():
    with pytest.raises(ValueError, match="margin_halfwidth"):
        Priors(0.6)
    with pytest.raises(ValueError, match="sigma_range"):
        Priors(sigma_range=(0.2, 1.2))
    with pytest.raises(ValueError, match="negative"):
        Priors(market_elasticity_range=(-1.0, 0.5))


UNEQUAL_PRICES = np.array([1.6, 1.3, 1.0])


def unequal_price_market():
    return Market.build(
        UNEQUAL_PRICES,
        Shares.total([0.30, 0.20, 0.35], "revenue"),
        Margins.lerner([0.4, np.nan, np.nan]),
        ["A", "B", "C"],
        revenue=500.0,
        outside_price=1.0,
    )


@pytest.mark.parametrize(
    "spec",
    [
        ModelSpec("logit"),
        ModelSpec("ces"),
        ModelSpec("pcaids", own_elasticity=-2.5, market_elasticity=-1.2),
        ModelSpec("linear", fill_margins_from="logit"),
    ],
    ids=lambda s: s.kind,
)
def test_unequal_prices_do_not_break_the_metrics(spec):
    """CES, PCAIDS and linear demand can imply quantity diversions summing above one."""
    mk = unequal_price_market()
    out = simulate_one(spec, mk, mk.ownership.merged(["A", "B"]))
    assert out.status == "ok", out.message
    assert out.guppi is not None and np.all(np.isfinite(out.guppi)) and out.guppi[0] > 0


def test_ces_at_unequal_prices_has_quantity_diversion_row_sums_above_one():
    mk = Market.build(
        np.array([1.0, 3.0, 0.5]),
        Shares.total([0.3, 0.2, 0.35], "revenue"),
        Margins.lerner([0.4, np.nan, np.nan]),
        ["A", "B", "C"],
        outside_price=1.0,
    )
    post = mk.ownership.merged(["A", "B"])
    out = simulate_one(ModelSpec("ces"), mk, post)
    assert out.status == "ok", out.message
    assert out.calibration is not None
    d = out.calibration.demand.diversion_quantity(mk.prices)
    assert d.sum(axis=1).max() > 1.0


def test_metric_failures_are_returned_as_outcomes(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr("mergerlab.simulate.guppi", broken)
    mk = heinz_like()
    out = simulate_one(ModelSpec("logit"), mk, mk.ownership.merged(["Heinz", "Beech-Nut"]))
    assert out.status == "metrics_failed"
    assert "synthetic failure" in (out.message or "")
    assert out.prices_post is not None and out.price_change is not None


def test_party_price_change_uses_revenue_weights_at_unequal_prices():
    mk = unequal_price_market()
    post = mk.ownership.merged(["A", "B"])
    res = simulate_merger(mk, post, [ModelSpec("logit")], efficiency=0.03)
    out = res.outcomes[0]
    q0 = out.calibration.demand.quantities(mk.prices)
    idx = [0, 1]
    hand = np.sum(mk.prices[idx] * q0[idx] * out.price_change[idx]) / np.sum(
        mk.prices[idx] * q0[idx]
    )
    assert res.party_price_change(out) == pytest.approx(hand, rel=1e-12)
    q_weighted = np.sum(q0[idx] * out.price_change[idx]) / np.sum(q0[idx])
    assert abs(hand - q_weighted) > 1e-4


def test_parties_surplus_change_sums_the_merging_products_only():
    mk = unequal_price_market()
    post = mk.ownership.merged(["A", "B"])
    out = simulate_merger(mk, post, [ModelSpec("logit")], efficiency=0.03).outcomes[0]
    demand, c0 = out.calibration.demand, out.calibration.costs
    c1 = c0.copy()
    c1[[0, 1]] *= 1.0 - 0.03
    ps0 = (mk.prices - c0) * demand.quantities(mk.prices)
    ps1 = (out.prices_post - c1) * demand.quantities(out.prices_post)
    assert out.parties_surplus_change == pytest.approx(ps1[[0, 1]].sum() - ps0[[0, 1]].sum())
    assert out.producer_surplus_change == pytest.approx(ps1.sum() - ps0.sum())
    assert abs(out.parties_surplus_change - out.producer_surplus_change) > 1e-6


@pytest.mark.parametrize("kind", ["logit", "ces", "pcaids", "linear"])
def test_monte_carlo_with_zero_width_priors_matches_the_simulation_at_unequal_prices(kind):
    """Party weights p q (not q) and the quantity-basis default diversion in the MC."""
    mk = unequal_price_market()
    post = mk.ownership.merged(["A", "B"])
    spec = {
        "pcaids": ModelSpec("pcaids", own_elasticity=-2.5, market_elasticity=-1.2),
        "linear": ModelSpec("linear", fill_margins_from="logit"),
    }.get(kind, ModelSpec(kind))
    res = simulate_merger(mk, post, [spec])
    mc = run_monte_carlo(
        mk, post, [spec], Priors(0.0, diversion_concentration=1e12), draws=3, seed=2
    )
    assert mc.counts(kind)["ok"] == 3
    np.testing.assert_allclose(
        mc.party_change[kind], res.party_price_change(res.outcomes[0]), rtol=1e-4
    )
    cm = res.outcomes[0].cmcr.relative
    q0 = res.outcomes[0].calibration.demand.quantities(mk.prices)
    w = mk.prices * q0
    hand = np.sum(w[[0, 1]] * cm[[0, 1]]) / np.sum(w[[0, 1]])
    np.testing.assert_allclose(mc.party_cmcr[kind], hand, rtol=1e-4)


def test_monte_carlo_records_the_parameter_range_of_successful_draws():
    mk = heinz_like()
    post = mk.ownership.merged(["Heinz", "Beech-Nut"])
    pri = Priors(0.10, (0.1, 0.7), (-1.5, -0.5), 30.0)
    mc = run_monte_carlo(mk, post, SPECS[:2], pri, draws=200, seed=3)
    lo, hi = mc.covered_range("nested_logit", "sigma")
    assert 0.1 <= lo < hi <= 0.7
    ok = np.array([st == "ok" for st in mc.status["nested_logit"]])
    assert hi == pytest.approx(mc.parameter_draws["sigma"][ok].max())
    assert hi < 0.7 - 1e-3  # failures concentrate at high nesting parameters
    assert mc.covered_range("logit", "sigma") is not None
