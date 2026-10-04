"""Unilateral-effects metrics: GUPPI, UPP, CMCR, first-order approximation, HM test, welfare."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .demand.base import Demand
from .supply import _omega, foc, markup_matrix, solve_bertrand
from .units import Diversion, DiversionBasis, FloatArray, Margins, Ownership, check_prices

# --------------------------------------------------------------------------- GUPPI / UPP


def guppi_matrix(prices: ArrayLike, margins: Margins, diversion: Diversion) -> FloatArray:
    """Pairwise GUPPI ``G[i, j] = D_ij m_j p_j / p_i`` (Farrell and Shapiro 2010; Moresi 2010).

    ``D`` must be a quantity diversion and ``m`` Lerner margins: with revenue or value
    diversions the price ratio would be applied twice.
    """
    p = check_prices(prices)
    d = diversion.require_quantity("GUPPI")
    m = margins.as_lerner(p)
    if d.shape != (p.size, p.size) or m.size != p.size:
        raise ValueError("prices, margins and diversion must describe the same products")
    return d * (m * p)[None, :] / p[:, None]


def guppi(
    prices: ArrayLike,
    margins: Margins,
    diversion: Diversion,
    pre: Ownership,
    post: Ownership,
) -> FloatArray:
    """Gross upward pricing pressure index of each product, as a fraction of its price.

    ``GUPPI_i = sum_j (Omega_post - Omega_pre)_ij D_ij m_j p_j / p_i``. For two
    single-product firms this is ``D_12 m_2 p_2 / p_1`` for product 1.
    """
    g = guppi_matrix(prices, margins, diversion)
    delta = post.matrix - pre.matrix
    return np.sum(delta * g, axis=1)


def upp(
    guppi_values: ArrayLike, margins: Margins, prices: ArrayLike, efficiency: ArrayLike
) -> FloatArray:
    """Upward pricing pressure net of efficiencies, as a fraction of price.

    ``UPP_i / p_i = GUPPI_i - e_i (1 - m_i)`` with ``e_i`` the proportional marginal-cost
    saving of product i.
    """
    p = check_prices(prices)
    g = np.asarray(guppi_values, dtype=float)
    e = np.broadcast_to(np.asarray(efficiency, dtype=float), g.shape)
    return g - e * (1.0 - margins.as_lerner(p))


def critical_efficiency(guppi_values: ArrayLike, margins: Margins, prices: ArrayLike) -> FloatArray:
    """Proportional cost saving ``e* = GUPPI / (1 - m)`` that sets UPP to zero."""
    p = check_prices(prices)
    return np.asarray(guppi_values, dtype=float) / (1.0 - margins.as_lerner(p))


def pricing_pressure(
    demand: Demand,
    p: FloatArray,
    costs: FloatArray,
    pre: Ownership,
    post: Ownership,
) -> FloatArray:
    """Merger pricing pressure ``g`` of Jaffe and Weyl (2013) in price units.

    ``g = -Delta_pre^{-1} ((Omega_post - Omega_pre) * J') (p - c)`` with
    ``Delta_pre = Omega_pre * J'``; for two single-product firms ``g_1 = D_12 (p_2 - c_2)``.
    """
    jac = demand.jacobian(p)
    d_pre = pre.matrix * jac.T
    x = (post.matrix - pre.matrix) * jac.T
    return -np.linalg.solve(d_pre, x @ (p - costs))


def merger_pass_through(
    demand: Demand,
    p: FloatArray,
    costs: FloatArray,
    pre: Ownership,
    post: Ownership,
    log_prices: bool = False,
) -> FloatArray:
    """Merger pass-through matrix ``rho = -(dh/dp)^{-1}`` (Jaffe and Weyl 2013, eq. 4).

    ``h(p) = -Delta_pre(p)^{-1} F_post(p)`` is the post-merger first-order condition in
    markup form (``F_post = q + (Omega_post * J')(p - c)``). Row j of ``rho`` gives the
    response of price j to a unit pricing-pressure shock to each product. The derivative
    is exact (complex step).

    With ``log_prices=True`` each condition is divided by its price and differentiated with
    respect to ``ln p``: ``M = -(d (h / p) / d ln p)^{-1}``, so that ``dp / p = M GUPPI``. This
    is the form Koh (2025, footnote 7) uses for CES demand.
    """
    step = 1e-30
    n = p.size
    om_pre, om_post = pre.matrix, post.matrix

    def h(x: FloatArray) -> FloatArray:
        jac = demand.jacobian(x)
        f_post = demand.quantities(x) + (om_post * jac.T) @ (x - costs)
        out = -np.linalg.solve(om_pre * jac.T, f_post)
        return out / x if log_prices else out

    dh = np.empty((n, n))
    pc = p.astype(complex)
    for k in range(n):
        e = np.zeros(n, dtype=complex)
        e[k] = 1j * step
        shifted = pc * np.exp(e) if log_prices else pc + e
        dh[:, k] = h(shifted).imag / step
    return -np.linalg.inv(dh)


def cost_pressure(
    demand: Demand,
    p: FloatArray,
    pre: Ownership,
    post: Ownership,
    cost_change: FloatArray,
) -> FloatArray:
    """Pricing pressure created by a marginal-cost change ``dc`` at the pre-merger prices.

    A cost change shifts the post-merger first-order condition by ``-(Omega_post * J') dc``,
    so in the markup form of :func:`pricing_pressure` it adds ``Delta_pre^{-1} (Omega_post *
    J') dc``. Row j is ``dc_j - sum_k D_jk dc_k`` over the products k that j's owner also
    prices after the merger: a saving on one merging product also lowers the opportunity
    cost of the other one and so is not passed on one for one.
    """
    dc = np.asarray(cost_change, dtype=float)
    jac = demand.jacobian(p)
    return np.linalg.solve(pre.matrix * jac.T, (post.matrix * jac.T) @ dc)


def first_order_price_effects(
    demand: Demand,
    p: FloatArray,
    costs: FloatArray,
    pre: Ownership,
    post: Ownership,
    cost_change: FloatArray | None = None,
    log_prices: bool = False,
) -> FloatArray:
    """First-order approximation of the post-merger price change (Jaffe and Weyl 2013).

    ``dp = rho (g + t)`` with ``g`` the merger pricing pressure and ``t`` the pressure from
    the marginal-cost change ``dc`` (negative for efficiencies), see :func:`cost_pressure`;
    both are evaluated at the pre-merger prices, so the approximation is exact to first
    order in the merger and in ``dc``. With ``log_prices=True`` the log-price form is used,
    ``dp / p = M (g + t) / p``; the result is again in currency units.
    """
    g = pricing_pressure(demand, p, costs, pre, post)
    if cost_change is not None:
        g = g + cost_pressure(demand, p, pre, post, cost_change)
    m = merger_pass_through(demand, p, costs, pre, post, log_prices)
    return p * (m @ (g / p)) if log_prices else m @ g


# ------------------------------------------------------------------------------- CMCR


def cmcr_two_product(
    m1: float, m2: float, d12: float, d21: float, price_ratio: float = 1.0
) -> float:
    """Werden's (1996) compensating marginal cost reduction of product 1, closed form.

    ``CMCR_1 = [m1 D12 D21 + m2 D12 (p2/p1)] / [(1 - m1)(1 - D12 D21)]`` as a fraction of
    product 1's marginal cost, with Lerner margins ``m`` and *quantity* diversions ``D``.
    ``price_ratio`` is ``p2 / p1``.
    """
    for name, v in (("m1", m1), ("m2", m2)):
        if not 0 < v < 1:
            raise ValueError(f"{name} must be a Lerner margin in (0, 1)")
    if not (0 <= d12 <= 1 and 0 <= d21 <= 1) or d12 * d21 >= 1:
        raise ValueError("diversion ratios must lie in [0, 1] with D12 D21 < 1")
    return (m1 * d12 * d21 + m2 * d12 * price_ratio) / ((1.0 - m1) * (1.0 - d12 * d21))


@dataclass(frozen=True)
class CMCR:
    """Compensating marginal cost reduction for each product (zero for non-parties)."""

    relative: FloatArray
    level: FloatArray
    costs_neutral: FloatArray


def cmcr_from_diversion(
    prices: ArrayLike,
    margins: Margins,
    diversion: Diversion,
    pre: Ownership,
    post: Ownership,
) -> CMCR:
    """Werden's CMCR in general matrix form from prices, margins and quantity diversions.

    At the pre-merger prices the first-order condition of product j is
    ``a_j - sum_{k != j} Omega_jk D_jk a_k = q_j / |dq_j/dp_j|`` with ``a = p - c``; the
    right side does not depend on ownership. Solving the post-merger system for the margin
    vector that keeps prices unchanged gives the cost reduction.
    """
    p = check_prices(prices)
    d = diversion.require_quantity("CMCR")
    a = margins.as_absolute(p)
    if np.any(~np.isfinite(a)):
        raise ValueError("CMCR needs a margin for every product")
    n = p.size
    if d.shape != (n, n):
        raise ValueError("prices, margins and diversion must describe the same products")
    a_pre = np.eye(n) - pre.matrix * d
    a_post = np.eye(n) - post.matrix * d
    a_new = np.linalg.solve(a_post, a_pre @ a)
    costs = p - a
    level = a_new - a
    return CMCR(level / costs, level, costs - level)


def cmcr_from_demand(
    demand: Demand,
    p: FloatArray,
    costs: FloatArray,
    post: Ownership,
) -> CMCR:
    """CMCR from a calibrated demand system: costs that make ``p`` a post-merger equilibrium.

    ``c' = p + (Omega_post * J')^{-1} q``; the reduction is ``c - c'``.
    """
    q = demand.quantities(p)
    c_new = p + np.linalg.solve(markup_matrix(demand, p, post), q)
    level = costs - c_new
    return CMCR(level / costs, level, c_new)


def synergy_amounts(cmcr: CMCR, quantities: ArrayLike) -> tuple[FloatArray, float]:
    """Cost synergies per unit and in total implied by a CMCR.

    ``cmcr.level`` is in currency per unit; multiplied by the units sold at pre-merger
    prices it gives the annual (or per-period) amount the merging products must save.
    """
    q = np.asarray(quantities, dtype=float)
    if q.shape != cmcr.level.shape:
        raise ValueError("quantities must have one entry per product")
    return cmcr.level, float(np.sum(cmcr.level * q))


def diversion_from_demand(
    demand: Demand, p: FloatArray, basis: DiversionBasis = DiversionBasis.QUANTITY
) -> Diversion:
    """Diversion ratios implied by a demand system at prices ``p``, on the requested basis.

    ``QUANTITY``: ``-(dq_j/dp_i) / (dq_i/dp_i)``. ``VALUE``: times ``p_j / p_i``.
    ``REVENUE``: ``-(d r_j / d p_i) / (d r_i / d p_i)`` with ``r = p q``. ``SHARE``:
    ``-(d w_j / d p_i) / (d w_i / d p_i)`` with ``w = r / sum(r)``, the budget-share slope
    ratio that AIDS-type calibrations report.
    """
    q = Diversion.from_demand(demand.diversion_quantity(p))
    if basis is DiversionBasis.QUANTITY:
        return q
    if basis is DiversionBasis.VALUE:
        return q.to_value(p)
    if basis is DiversionBasis.REVENUE:
        return q.to_revenue(p, np.diag(demand.elasticities(p)))
    jac = demand.jacobian(p)
    qty = demand.quantities(p)
    rev = p * qty
    total = rev.sum()
    w = rev / total
    d_rev = np.diag(qty) + p[:, None] * jac  # d r_j / d p_i
    dw = (d_rev - w[:, None] * d_rev.sum(axis=0)[None, :]) / total
    share = -dw.T / np.diag(dw)[:, None]
    return Diversion.share(share)


# -------------------------------------------------------------- hypothetical monopolist


class _Restricted(Demand):
    """Demand of a product subset with the other prices held fixed."""

    name = "restricted"

    def __init__(self, base: Demand, idx: np.ndarray, fixed: FloatArray) -> None:
        self.base, self.idx, self.fixed = base, idx, fixed

    @property
    def n_products(self) -> int:
        return int(self.idx.size)

    def _embed(self, p: FloatArray) -> FloatArray:
        full = np.array(self.fixed, dtype=p.dtype)
        full[self.idx] = p
        return full

    def quantities(self, p: FloatArray) -> FloatArray:
        return self.base.quantities(self._embed(p))[self.idx]

    def jacobian(self, p: FloatArray) -> FloatArray:
        return self.base.jacobian(self._embed(p))[np.ix_(self.idx, self.idx)]

    def consumer_surplus(self, p: FloatArray) -> float:
        return self.base.consumer_surplus(self._embed(p))

    def is_valid(self, p: FloatArray) -> bool:
        return self.base.is_valid(self._embed(np.asarray(p)))


@dataclass(frozen=True)
class HMTest:
    """Hypothetical monopolist test for a candidate market."""

    products: tuple[int, ...]
    ssnip: float
    price_increase: FloatArray
    passes: bool
    ssnip_profitable: bool
    critical_loss: float
    actual_loss: float
    max_increase: float


def critical_loss(ssnip: float, lerner_margin: float) -> float:
    """Critical loss ``t / (t + m)``: the largest unit loss a SSNIP ``t`` can sustain."""
    if not ssnip > 0:
        raise ValueError("ssnip must be positive")
    if not 0 < lerner_margin < 1:
        raise ValueError("lerner_margin must lie in (0, 1)")
    return ssnip / (ssnip + lerner_margin)


def hypothetical_monopolist_test(
    demand: Demand,
    p: FloatArray,
    costs: FloatArray,
    products: ArrayLike,
    ssnip: float = 0.05,
    rivals: str = "fixed",
    ownership: Ownership | None = None,
) -> HMTest:
    """Hypothetical monopolist test by simulation.

    The hypothetical monopolist owns ``products`` and sets their profit-maximising
    prices (solver gate enforced). ``rivals="fixed"`` holds the other products' prices
    constant (as in the 2010 HMG, section 4.1); ``"respond"`` lets them re-optimise under
    ``ownership`` (pre-merger ownership by default: separate owners). The candidate
    market passes if the monopolist's optimum raises the price of at least one product
    by at least ``ssnip``.

    Also reported: whether a uniform SSNIP is profitable, and the critical-loss shortcut
    ``t / (t + m)`` with the revenue-weighted margin against the revenue-weighted unit
    loss from the uniform SSNIP (exact for equal margins).
    """
    idx = np.unique(np.asarray(products, dtype=int))
    n = p.size
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
        raise ValueError("products must index products of the demand system")
    if rivals not in ("fixed", "respond"):
        raise ValueError("rivals must be 'fixed' or 'respond'")
    if not ssnip > 0:
        raise ValueError("ssnip must be positive")
    if rivals == "fixed":
        sub = _Restricted(demand, idx, p)
        eq = solve_bertrand(sub, costs[idx], np.ones((idx.size, idx.size)), p[idx])
        p_new = p.copy()
        p_new[idx] = eq.prices
    else:
        omega = np.eye(n) if ownership is None else ownership.matrix.copy()
        omega[np.ix_(idx, idx)] = 1.0
        eq = solve_bertrand(demand, costs, omega, p)
        p_new = eq.prices
    increase = p_new[idx] / p[idx] - 1.0
    q0 = demand.quantities(p)
    p_s = p.copy()
    p_s[idx] = p[idx] * (1.0 + ssnip)
    q1 = demand.quantities(p_s)
    profit0 = float(((p - costs) * q0)[idx].sum())
    profit1 = float(((p_s - costs) * q1)[idx].sum())
    rev = (p * q0)[idx]
    margin = float(((p - costs) * q0)[idx].sum() / rev.sum())
    loss = float((rev * (1.0 - q1[idx] / q0[idx])).sum() / rev.sum())
    return HMTest(
        tuple(int(i) for i in idx),
        ssnip,
        increase,
        bool(increase.max() >= ssnip - 1e-12),
        profit1 > profit0,
        critical_loss(ssnip, margin),
        loss,
        float(increase.max()),
    )


# ------------------------------------------------------------------------------ welfare


def producer_surplus(demand: Demand, p: FloatArray, costs: FloatArray) -> FloatArray:
    """Variable profit ``(p_j - c_j) q_j`` of each product."""
    return (p - costs) * demand.quantities(p)


def compensating_variation(demand: Demand, p0: FloatArray, p1: FloatArray) -> float:
    """Money that consumers must receive to be as well off at ``p1`` as at ``p0``.

    Equals ``CS(p0) - CS(p1)`` for the demand system's surplus function: closed form for
    logit, nested logit and CES, a Marshallian path integral for linear demand and PCAIDS
    (exact when income effects are absent).
    """
    return demand.consumer_surplus(p0) - demand.consumer_surplus(p1)


def foc_residual(
    demand: Demand, p: FloatArray, costs: FloatArray, ownership: Ownership | FloatArray
) -> float:
    """Scaled first-order-condition residual ``max_j |F_j| / q_j``."""
    q = demand.quantities(p)
    return float(np.max(np.abs(foc(demand, p, costs, _omega(ownership))) / q))
