import numpy as np
import pytest

from helpers import naive_fixed_point, random_logit_market
from mergerlab.demand import Linear, Logit, NestedLogit, calibrate_logit
from mergerlab.errors import CalibrationError, EquilibriumNotFound
from mergerlab.supply import (
    _zeta_iterate,
    foc,
    foc_jacobian,
    recover_costs,
    scaled_residual,
    solve_bertrand,
)
from mergerlab.units import Ownership


def _demand_from_scenario(sc):
    delta = np.array(sc["delta"])
    if sc["sigma"] is None:
        return Logit(delta, sc["alpha"])
    return NestedLogit(delta, sc["alpha"], sc["sigma"], sc["nests"])


def _scenario_ids(fixtures):
    return [s["name"] for s in fixtures["scenarios"]]


def test_pyblp_oracle_pre_and_post_merger_prices(pyblp_fixtures):
    """Logit and nested-logit equilibria agree with pyblp to near machine precision."""
    for sc in pyblp_fixtures["scenarios"]:
        demand = _demand_from_scenario(sc)
        costs = np.array(sc["costs"])
        pre = Ownership.from_owners(sc["owners_pre"])
        post = Ownership.from_owners(sc["owners_post"])
        eq_pre = solve_bertrand(demand, costs, pre, costs + 1.0)
        eq_post = solve_bertrand(demand, costs, post, eq_pre.prices)
        np.testing.assert_allclose(
            eq_pre.prices, sc["pre"]["prices"], rtol=1e-11, err_msg=sc["name"]
        )
        np.testing.assert_allclose(
            eq_post.prices, sc["post"]["prices"], rtol=1e-11, err_msg=sc["name"]
        )
        np.testing.assert_allclose(eq_post.quantities, sc["post"]["shares"], rtol=1e-10)
        assert eq_post.residual < 1e-10


def test_naive_fixed_point_fails_where_the_gated_solver_succeeds(pyblp_fixtures):
    """Regression: the textbook iteration returns a non-equilibrium for nested logit.

    On the sigma = 0.6 scenario the plain iteration does not converge and reports a
    price increase of about +149% where pyblp (and this package) find +43.5%.
    """
    sc = next(s for s in pyblp_fixtures["scenarios"] if s["name"].startswith("nested_0.6"))
    demand = _demand_from_scenario(sc)
    costs = np.array(sc["costs"])
    post = Ownership.from_owners(sc["owners_post"])
    pre_prices = np.array(sc["pre"]["prices"])
    naive = naive_fixed_point(demand, costs, post, costs + 1.0)
    assert scaled_residual(demand, naive, costs, post) > 1.0
    assert (naive / pre_prices - 1).max() > 1.2
    eq = solve_bertrand(demand, costs, post, pre_prices)
    effect = (eq.prices / pre_prices - 1).max()
    assert effect == pytest.approx(0.43486125, abs=1e-7)
    assert eq.residual < 1e-10
    pyblp_effect = (np.array(sc["post"]["prices"]) / pre_prices - 1).max()
    assert effect == pytest.approx(pyblp_effect, rel=1e-10)


def test_recover_costs_makes_observed_prices_an_equilibrium():
    rng = np.random.default_rng(4)
    mk = random_logit_market(rng, 5, firms=3)
    cal = calibrate_logit(mk)
    assert scaled_residual(cal.demand, mk.prices, cal.costs, mk.ownership) < 1e-13
    again = recover_costs(cal.demand, mk.prices, mk.ownership)
    np.testing.assert_allclose(again, cal.costs)


def test_complex_step_jacobian_matches_finite_differences():
    demand = NestedLogit(np.array([1.0, 1.4, 0.8, 1.2]), 1.6, 0.5, [0, 0, 1, 1], 2.0)
    own = Ownership.from_owners(["A", "A", "B", "C"])
    costs = np.array([0.4, 0.5, 0.45, 0.6])
    p = np.array([1.0, 1.1, 1.05, 1.2])
    jac = foc_jacobian(demand, p, costs, own)
    num = np.empty((4, 4))
    for k in range(4):
        e = np.zeros(4)
        e[k] = 1e-6
        num[:, k] = (
            foc(demand, p + e, costs, own.matrix) - foc(demand, p - e, costs, own.matrix)
        ) / 2e-6
    np.testing.assert_allclose(jac, num, rtol=1e-6, atol=1e-8)


def test_solution_does_not_depend_on_the_start():
    demand = Logit(np.array([1.0, 1.4, 0.8, 1.2]), 1.6, 2.0)
    own = Ownership.from_owners(["A", "A", "B", "C"])
    costs = np.array([0.4, 0.5, 0.45, 0.6])
    ref = solve_bertrand(demand, costs, own, costs + 1.0).prices
    for start in (costs * 1.01, costs + 5.0, costs + 0.05):
        np.testing.assert_allclose(
            solve_bertrand(demand, costs, own, start).prices, ref, rtol=1e-11
        )


def test_market_size_does_not_change_prices():
    delta = np.array([1.0, 1.4, 0.8])
    own = Ownership.from_owners(["A", "A", "B"])
    costs = np.array([0.4, 0.5, 0.45])
    a = solve_bertrand(Logit(delta, 1.6, 1.0), costs, own, costs + 1).prices
    b = solve_bertrand(Logit(delta, 1.6, 250.0), costs, own, costs + 1).prices
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_gate_raises_with_diagnostics_when_tolerance_is_unreachable():
    demand = Logit(np.array([1.0, 1.4]), 1.6, 1.0)
    own = Ownership.from_owners(["A", "B"])
    costs = np.array([0.4, 0.5])
    with pytest.raises(EquilibriumNotFound) as err:
        solve_bertrand(demand, costs, own, costs + 1.0, tol=1e-30)
    assert err.value.diagnostics["gate"] == 1e-30
    assert err.value.diagnostics["attempts"]
    assert "Bertrand-Nash" in str(err.value)


def test_gate_raises_when_no_valid_equilibrium_exists():
    """Linear demand q = 1 - p with marginal cost above the choke price."""
    demand = Linear(np.array([1.0]), np.array([[-1.0]]), np.array([0.5]))
    with pytest.raises(EquilibriumNotFound):
        solve_bertrand(demand, np.array([1.5]), np.ones((1, 1)), np.array([0.6]))


def test_linear_demand_equilibrium_is_closed_form():
    """Single product q = a - b p: p* = (a / b + c) / 2."""
    a, b, c = 4.0, 2.0, 0.6
    demand = Linear(np.array([a]), np.array([[-b]]), np.array([1.0]))
    eq = solve_bertrand(demand, np.array([c]), np.ones((1, 1)), np.array([1.0]))
    assert eq.prices[0] == pytest.approx((a / b + c) / 2, rel=1e-14)


def test_returned_solutions_always_pass_the_gate_on_random_markets():
    rng = np.random.default_rng(11)
    done = 0
    while done < 25:
        mk = random_logit_market(rng, int(rng.integers(3, 8)), firms=None)
        try:
            cal = calibrate_logit(mk)
        except CalibrationError:
            continue
        post = mk.ownership.merged(["F0", "F1"])
        eq = solve_bertrand(cal.demand, cal.costs, post, mk.prices)
        assert scaled_residual(cal.demand, eq.prices, cal.costs, post) < 1e-10
        assert np.all(eq.prices >= mk.prices - 1e-12)
        done += 1


def test_scaled_residual_is_the_largest_violation_relative_to_own_sales():
    demand = Logit(np.array([1.5, 1.0, 0.8]), 1.6, 1.0)
    costs = np.array([0.4, 0.5, 0.45])
    own = Ownership.from_owners(["A", "B", "C"])
    eq = solve_bertrand(demand, costs, own, costs + 1.0)
    assert scaled_residual(demand, eq.prices, costs, own) < 1e-10
    p = eq.prices.copy()
    p[1] += 0.05  # moves every first-order condition, product 1's the most
    f = foc(demand, p, costs, own.matrix)
    ratios = np.abs(f) / demand.quantities(p)
    assert ratios.max() > ratios.mean() + 1e-3
    assert scaled_residual(demand, p, costs, own) == pytest.approx(ratios.max(), rel=1e-12)


def test_recover_costs_rejects_prices_below_implied_costs():
    """Upward-sloping own demand implies a negative margin; that is not Bertrand pricing."""
    demand = Linear(np.array([1.0, 6.0]), np.array([[0.5, 0.0], [0.0, -1.0]]), np.ones(2) * 4)
    own = Ownership.from_owners(["A", "B"])
    p = np.array([4.0, 4.0])
    margins_second = -np.linalg.solve(own.matrix * demand.jacobian(p).T, demand.quantities(p))[1]
    assert 0 < margins_second / p[1] < 1  # the second product alone would be fine
    with pytest.raises(CalibrationError, match="Lerner margins outside"):
        recover_costs(demand, p, own)


def test_zeta_iteration_alone_reaches_the_pyblp_prices_before_and_after_the_merger(
    pyblp_fixtures,
):
    """The Morrow-Skerlos fixed point converges without the Newton polish, including for
    nested logit (non-symmetric Gamma), and post-merger ownership exercises Gamma' itself."""
    nested = [s for s in pyblp_fixtures["scenarios"] if s["sigma"] is not None]
    assert nested
    for sc in nested:
        demand = _demand_from_scenario(sc)
        costs = np.array(sc["costs"])
        pre, post = sc["pre"]["prices"], np.array(sc["post"]["prices"])
        for owners, target in ((sc["owners_pre"], pre), (sc["owners_post"], post)):
            omega = Ownership.from_owners(owners).matrix
            p, iterations = _zeta_iterate(demand, costs + 1.0, costs, omega, 5000)
            np.testing.assert_allclose(p, target, rtol=1e-9, err_msg=sc["name"])
            assert iterations < 5000
        omega_post = Ownership.from_owners(sc["owners_post"]).matrix
        p_step, _ = _zeta_iterate(demand, post, costs, omega_post, 1)
        np.testing.assert_allclose(p_step, post, rtol=1e-12, err_msg=sc["name"])


class _AsymmetricLinear(Linear):
    """Linear demand with an asymmetric slope matrix and an explicit zeta decomposition."""

    supports_zeta = True

    def zeta_terms(self, p):
        lam = -np.diag(self.slopes).copy()
        return lam, self.slopes + np.diag(lam)


def test_zeta_step_fixes_the_equilibrium_only_with_the_transposed_gamma():
    """Logit-type systems have symmetric Gamma, so the transpose is only visible with an
    asymmetric one. The closed-form linear-demand equilibrium must be a fixed point."""
    slopes = np.array([[-2.0, 0.9, 0.1], [0.2, -1.8, 0.7], [0.5, 0.1, -2.2]])
    intercept = np.array([3.0, 2.6, 3.1])
    costs = np.array([0.4, 0.5, 0.45])
    demand = _AsymmetricLinear(intercept, slopes, np.ones(3))
    omega = Ownership.from_owners(["A", "A", "B"]).matrix
    delta = omega * slopes.T
    expected = -np.linalg.solve(slopes + delta, intercept - delta @ costs)
    assert np.all(expected > costs)
    assert scaled_residual(demand, expected, costs, omega) < 1e-13
    p, _ = _zeta_iterate(demand, expected, costs, omega, 1)
    np.testing.assert_allclose(p, expected, rtol=1e-12)
