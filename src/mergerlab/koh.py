"""Concentration-based inference: consumer harm from the change in HHI (Koh 2025).

For a merger of firms A and B without synergies, the first-order approximation of Jaffe and
Weyl (2013) to the change in consumer surplus is ``dCS = -dp'q`` with ``dp = M g``, where ``g``
is the pricing pressure and ``M`` the merger pass-through matrix. For logit and CES demand this
factorises (Koh 2025, Proposition 1 and Remark 1):

    dCS = -V0 * rho1 * rho2 * dHHI,       dHHI = 2 s_A s_B,

    V0   = N / alpha (logit)      or  Y / (sigma - 1) (CES),
    phi  = 1 (logit)              or  sigma / (sigma - 1) (CES),
    rho1 = phi / ((phi - s_A) (phi - s_B)),

with ``s`` the quantity shares of the total market (logit) or revenue shares of income (CES).
``rho2`` collects the merger pass-through matrix; it equals one when ``M = phi I``, which holds
approximately for small shares. For single-product firms at unit prices

    rho2 = (M_AA + M_BB + M_AB s_A / s_B + M_BA s_B / s_A) / (2 phi).

This module computes the decomposition from a calibrated demand system. The formula is a
first-order statement about the merging products with the rivals' prices held fixed, so the
like-for-like benchmark is the exact compensating variation of the same price change
(:attr:`HarmComparison.exact_rivals_fixed`). The full simulation additionally lets the rivals
re-price; that response, not curvature, accounts for most of the gap between the formula and
the simulated harm in a three-to-two merger with a large rival
(:func:`harm_comparison` separates the two).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demand import CES, Demand, Logit
from .metrics import _Restricted, merger_pass_through, pricing_pressure
from .supply import solve_bertrand
from .units import FloatArray, Ownership

__all__ = [
    "HarmComparison",
    "KohDecomposition",
    "first_order_harm_error",
    "harm_comparison",
    "koh_decomposition",
    "koh_rho1",
]


def koh_rho1(phi: float, share_a: float, share_b: float) -> float:
    """``phi / ((phi - s_A)(phi - s_B))``: the cross-firm scaling factor of Koh (2025)."""
    if not (phi > share_a > 0 and phi > share_b > 0):
        raise ValueError("shares must be positive and smaller than phi")
    return phi / ((phi - share_a) * (phi - share_b))


@dataclass(frozen=True)
class KohDecomposition:
    """``harm = v0 * rho1 * rho2 * delta_hhi`` (harm is a positive loss in currency)."""

    model: str
    phi: float
    v0: float
    rho1: float
    rho2: float
    delta_hhi: float
    share_a: float
    share_b: float
    pass_through: FloatArray

    @property
    def rho(self) -> float:
        return self.v0 * self.rho1 * self.rho2

    @property
    def harm(self) -> float:
        return self.rho * self.delta_hhi

    @property
    def harm_with_rho2_one(self) -> float:
        """Harm under the shortcut ``rho2 = 1`` (``M = phi I``)."""
        return self.v0 * self.rho1 * self.delta_hhi


def _scale(demand: Demand, p: FloatArray) -> tuple[str, float, float, FloatArray]:
    if isinstance(demand, Logit):
        return "logit", 1.0, demand.market_size / demand.alpha, demand.shares(p)
    if isinstance(demand, CES):
        sigma = demand.sigma
        return "CES", sigma / (sigma - 1.0), demand.income / (sigma - 1.0), demand.revenue_shares(p)
    raise ValueError("the concentration-based formula is derived for logit and CES demand only")


def koh_decomposition(
    demand: Demand,
    prices: FloatArray,
    costs: FloatArray,
    firm_a: int,
    firm_b: int,
    rivals: str = "fixed",
) -> KohDecomposition:
    """Decompose the first-order consumer harm of a merger of two single-product firms.

    ``rivals="fixed"`` (Koh's Section 6) computes the pass-through matrix of the two merging
    products with all other prices held fixed; ``"respond"`` takes the merging block of the
    full merger pass-through matrix, which includes the feedback from rivals' price responses
    onto the merging products. In both cases the harm is the loss on the merging products, as
    in Koh's formula; rivals' own price changes are not counted. ``rho2`` is defined by
    ``harm = V0 rho1 rho2 dHHI`` with the first-order harm ``q'(M g)`` in price units, so it is
    exact for any prices, and equals the closed-form expression at unit prices. For CES demand
    the pass-through matrix is taken in log prices as in Koh (2025, footnote 7).
    """
    p = np.asarray(prices, dtype=float)
    c = np.asarray(costs, dtype=float)
    n = p.size
    if firm_a == firm_b or not (0 <= firm_a < n and 0 <= firm_b < n):
        raise ValueError("firm_a and firm_b must be two distinct product indices")
    if rivals not in ("fixed", "respond"):
        raise ValueError("rivals must be 'fixed' or 'respond'")
    model, phi, v0, s = _scale(demand, p)
    use_log = model == "CES"
    idx = np.array([firm_a, firm_b])
    q = demand.quantities(p)
    if rivals == "fixed":
        sub = _Restricted(demand, idx, p)
        pre, post = Ownership(np.eye(2)), Ownership(np.ones((2, 2)))
        g = pricing_pressure(sub, p[idx], c[idx], pre, post)
        m = merger_pass_through(sub, p[idx], c[idx], pre, post, log_prices=use_log)
        dp = p[idx] * (m @ (g / p[idx])) if use_log else m @ g
        first_order_harm = float(q[idx] @ dp)
    else:
        owners = [f"F{i}" for i in range(n)]
        pre = Ownership.from_owners(owners)
        post = pre.merged([owners[firm_a], owners[firm_b]])
        g = pricing_pressure(demand, p, c, pre, post)
        full = merger_pass_through(demand, p, c, pre, post, log_prices=use_log)
        m = full[np.ix_(idx, idx)]
        dp = p[idx] * (m @ (g[idx] / p[idx])) if use_log else m @ g[idx]
        first_order_harm = float(q[idx] @ dp)
    s_a, s_b = float(s[firm_a]), float(s[firm_b])
    rho1 = koh_rho1(phi, s_a, s_b)
    d_hhi = 2.0 * s_a * s_b
    rho2 = first_order_harm / (v0 * rho1 * d_hhi)
    return KohDecomposition(model, phi, v0, rho1, rho2, d_hhi, s_a, s_b, m)


@dataclass(frozen=True)
class HarmComparison:
    """Koh's first-order harm against exact harm of the same scope and of the full merger.

    All harms are positive losses in currency (compensating variation of all consumers).
    ``exact_rivals_fixed`` re-prices only the merging products to their post-merger optimum
    with the rivals' prices held fixed; ``full_simulation`` lets every price re-optimise.
    """

    first_order: float
    exact_rivals_fixed: float
    full_simulation: float
    price_change_rivals_fixed: FloatArray
    price_change_full: FloatArray

    @property
    def curvature_gap(self) -> float:
        """Exact minus first-order harm with rivals fixed: what the linearisation misses."""
        return self.exact_rivals_fixed - self.first_order

    @property
    def rival_response(self) -> float:
        """Harm added by the rivals' price responses (full minus rivals fixed)."""
        return self.full_simulation - self.exact_rivals_fixed

    @property
    def rival_response_share(self) -> float:
        """Share of the full simulated harm that comes from the rivals' price responses."""
        return self.rival_response / self.full_simulation


def harm_comparison(
    demand: Demand,
    prices: FloatArray,
    costs: FloatArray,
    firm_a: int,
    firm_b: int,
    rivals: str = "fixed",
) -> HarmComparison:
    """Compare the first-order harm with exact harm of the same scope and the full simulation.

    The three numbers separate the sources of a gap between Koh's formula and a merger
    simulation: curvature of demand (``exact_rivals_fixed - first_order``) and the price
    responses of firms outside the merger (``full_simulation - exact_rivals_fixed``).
    """
    dec = koh_decomposition(demand, prices, costs, firm_a, firm_b, rivals)
    p = np.asarray(prices, dtype=float)
    c = np.asarray(costs, dtype=float)
    n = p.size
    idx = np.array([firm_a, firm_b])
    cs0 = demand.consumer_surplus(p)
    sub = _Restricted(demand, idx, p)
    eq_fixed = solve_bertrand(sub, c[idx], np.ones((2, 2)), p[idx])
    p_fixed = p.copy()
    p_fixed[idx] = eq_fixed.prices
    owners = [f"F{i}" for i in range(n)]
    post = Ownership.from_owners(owners).merged([owners[firm_a], owners[firm_b]])
    eq_full = solve_bertrand(demand, c, post, p)
    return HarmComparison(
        dec.harm,
        float(cs0 - demand.consumer_surplus(p_fixed)),
        float(cs0 - demand.consumer_surplus(eq_full.prices)),
        p_fixed / p - 1.0,
        eq_full.prices / p - 1.0,
    )


def first_order_harm_error(
    demand: Demand,
    prices: FloatArray,
    costs: FloatArray,
    firm_a: int,
    firm_b: int,
    rivals: str = "fixed",
) -> tuple[float, float]:
    """First-order harm and the harm from the full merger simulation (compensating variation).

    Returns ``(first_order, simulated)``, both positive losses in currency. The two differ by
    curvature and by the rivals' price responses; :func:`harm_comparison` separates them.
    """
    cmp = harm_comparison(demand, prices, costs, firm_a, firm_b, rivals)
    return cmp.first_order, cmp.full_simulation
