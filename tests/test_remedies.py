"""Single-product divestiture evaluation."""

import numpy as np
import pytest

from mergerlab import (
    Margins,
    Market,
    ModelSpec,
    Ownership,
    Shares,
    evaluate_divestitures,
    simulate_one,
)


def two_by_two_market():
    """A owns a1, a2; B owns b1, b2; C is a fringe brand. Logit with a known outside share."""
    return Market.build(
        [1.0, 1.1, 0.9, 1.0, 1.0],
        Shares.total([0.18, 0.08, 0.20, 0.06, 0.28], "quantity"),
        Margins.lerner([0.35, np.nan, np.nan, np.nan, np.nan]),
        ["A", "A", "B", "B", "C"],
        labels=["a1", "a2", "b1", "b2", "c"],
    )


def test_every_divestiture_lowers_harm_and_options_are_ranked():
    mk = two_by_two_market()
    post = mk.ownership.merged(["A", "B"])
    report = evaluate_divestitures(ModelSpec("logit"), mk, post)
    base = report.baseline.consumer_harm
    assert base is not None and base > 0
    assert [o.label for o in report.options].count("a1") == 1
    harms = [o.consumer_harm for o in report.options]
    assert all(h is not None and 0 <= h < base for h in harms)
    assert harms == sorted(harms)
    assert report.best is report.options[0]
    for o in report.options:
        assert o.max_price_change is not None and o.max_price_change < max(
            report.baseline.price_change
        )


def test_divesting_the_closest_substitute_of_the_biggest_product_helps_most():
    mk = two_by_two_market()
    post = mk.ownership.merged(["A", "B"])
    report = evaluate_divestitures(ModelSpec("logit"), mk, post)
    # logit diversion is proportional to share, so the largest overlapping product is the
    # one whose removal from the merged firm's portfolio removes most internalised diversion
    assert report.best is not None and report.best.label in {"a1", "b1"}


def test_single_product_firms_are_fully_restored_by_any_divestiture():
    mk = Market.build(
        [1.0, 1.0, 1.0],
        Shares.total([0.3, 0.25, 0.25], "quantity"),
        Margins.lerner([0.4, np.nan, np.nan]),
        ["X", "Y", "Z"],
    )
    post = mk.ownership.merged(["X", "Y"])
    report = evaluate_divestitures(ModelSpec("logit"), mk, post)
    for o in report.options:
        np.testing.assert_allclose(o.outcome.prices_post, mk.prices, rtol=1e-10)
        assert o.consumer_harm == pytest.approx(0.0, abs=1e-10)


def test_restoring_pre_merger_ownership_and_matching_a_direct_simulation():
    mk = two_by_two_market()
    post = mk.ownership.merged(["A", "B"])
    out = simulate_one(ModelSpec("logit"), mk, mk.ownership)
    np.testing.assert_allclose(out.prices_post, mk.prices, rtol=1e-10)
    report = evaluate_divestitures(ModelSpec("logit"), mk, post, candidates=[1])
    owners = list(post.owners)
    owners[1] = "divested:a2"
    direct = simulate_one(ModelSpec("logit"), mk, Ownership.from_owners(owners))
    assert report.options[0].consumer_harm == pytest.approx(direct.consumer_harm)


def test_validation():
    mk = two_by_two_market()
    post = mk.ownership.merged(["A", "B"])
    with pytest.raises(ValueError, match="merging parties"):
        evaluate_divestitures(ModelSpec("logit"), mk, post, candidates=[4])
    with pytest.raises(ValueError, match="does not differ"):
        evaluate_divestitures(ModelSpec("logit"), mk, mk.ownership)
    with pytest.raises(ValueError, match="from_owners"):
        evaluate_divestitures(ModelSpec("logit"), mk, Ownership(np.eye(5)))
