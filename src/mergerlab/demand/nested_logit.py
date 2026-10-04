"""Nested logit demand with fixed nests and a single nesting parameter."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import CalibrationError
from ..market import Market
from ..supply import recover_costs
from ..units import FloatArray, ShareBasis
from .base import Calibration, Demand
from .logit import _fit_inverse_scale


def _nest_matrix(nests: Sequence[int]) -> FloatArray:
    labels = np.asarray(nests)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("nests must be a one-dimensional sequence of nest labels")
    uniq = np.unique(labels)
    return (labels[:, None] == uniq[None, :]).astype(float)


class NestedLogit(Demand):
    """Nested logit of Berry (1994) as parametrised by Bjornerstedt and Verboven (2016).

    Utility ``u_ij = delta_j - alpha p_j + zeta_ig + (1 - sigma) eps_ij``. Products in
    the same nest are closer substitutes the larger ``sigma`` in [0, 1); ``sigma = 0``
    is the plain logit. The outside good forms its own nest. With
    ``v_j = (delta_j - alpha p_j) / (1 - sigma)``, ``D_g = sum_{j in g} exp(v_j)``:

        s_{j|g} = exp(v_j) / D_g,   s_g = D_g^(1-sigma) / (1 + sum_h D_h^(1-sigma)),

    and ``q_j = M s_{j|g} s_g``.
    """

    name = "nested logit"
    supports_zeta = True

    def __init__(
        self,
        delta: FloatArray,
        alpha: float,
        sigma: float,
        nests: Sequence[int],
        market_size: float = 1.0,
    ) -> None:
        if not alpha > 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= sigma < 1.0:
            raise ValueError("the nesting parameter sigma must lie in [0, 1)")
        if not market_size > 0:
            raise ValueError("market_size must be positive")
        self.delta = np.asarray(delta, dtype=float)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.nests = tuple(int(n) for n in nests)
        self._g = _nest_matrix(self.nests)
        if self._g.shape[0] != self.delta.size:
            raise ValueError("nests and delta must have the same length")
        self._same = self._g @ self._g.T
        self.market_size = float(market_size)

    @property
    def n_products(self) -> int:
        return self.delta.size

    def _parts(self, p: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        v = (self.delta - self.alpha * p) / (1.0 - self.sigma)
        ev = np.exp(v)
        d_g = self._g.T @ ev
        s_in = ev / (self._g @ d_g)
        i_g = d_g ** (1.0 - self.sigma)
        s_g = i_g / (1.0 + i_g.sum())
        return s_in, s_g, i_g

    def shares(self, p: FloatArray) -> FloatArray:
        s_in, s_g, _ = self._parts(p)
        return s_in * (self._g @ s_g)

    def quantities(self, p: FloatArray) -> FloatArray:
        return self.market_size * self.shares(p)

    def zeta_terms(self, p: FloatArray) -> tuple[FloatArray, FloatArray]:
        s_in, s_g, _ = self._parts(p)
        s = s_in * (self._g @ s_g)
        m = self.market_size * self.alpha
        k = self.sigma / (1.0 - self.sigma)
        lam = m * s / (1.0 - self.sigma)
        gamma = m * s[:, None] * (k * self._same * s_in[None, :] + s[None, :])
        return lam, gamma

    def jacobian(self, p: FloatArray) -> FloatArray:
        lam, gamma = self.zeta_terms(p)
        return gamma - np.diag(lam)

    def consumer_surplus(self, p: FloatArray) -> float:
        """``(M / alpha) ln(1 + sum_g D_g^(1-sigma))``: McFadden's inclusive value."""
        _, _, i_g = self._parts(p)
        return float(self.market_size / self.alpha * np.log1p(i_g.sum()))


def calibrate_nested_logit(market: Market, nests: Sequence[int], sigma: float) -> Calibration:
    """Nested logit with known outside share, fixed nests and a given ``sigma``.

    Given shares and ``sigma`` the substitution matrix is proportional to ``alpha``, so
    ``alpha`` follows from the margins exactly as in the logit (least squares on Lerner
    margins when there are several). The mean utilities are
    ``delta_j - alpha p_j = ln(s_j / s_0) - sigma ln s_{j|g}``.
    """
    market.shares.require(ShareBasis.QUANTITY, True, "nested logit")
    s = market.shares.values
    s0 = float(market.shares.outside)  # type: ignore[arg-type]
    p = market.prices
    omega = market.ownership.matrix
    g = _nest_matrix(nests)
    if g.shape[0] != s.size:
        raise ValueError("nests must have one entry per product")
    if not 0.0 <= sigma < 1.0:
        raise ValueError("the nesting parameter sigma must lie in [0, 1)")
    nest_share = g.T @ s
    s_in = s / (g @ nest_share)
    same = g @ g.T
    k = sigma / (1.0 - sigma)
    lam = s / (1.0 - sigma)
    gamma = s[:, None] * (k * same * s_in[None, :] + s[None, :])
    jac_unit = gamma - np.diag(lam)
    try:
        markup_unit = -np.linalg.solve(omega * jac_unit.T, s)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("singular first-order conditions at the observed shares") from exc
    inv_alpha, resid = _fit_inverse_scale(markup_unit / p, market.lerner)
    alpha = 1.0 / inv_alpha
    mean_utility = np.log(s / s0) - sigma * np.log(s_in)
    delta = mean_utility + alpha * p
    q_inside = market.quantities()
    demand = NestedLogit(delta, alpha, sigma, nests, market_size=float(q_inside.sum()) / (1.0 - s0))
    costs = recover_costs(demand, p, omega)
    fitted = (p - costs) / p
    return Calibration(
        demand,
        market,
        costs,
        fitted,
        resid,
        {"alpha": alpha, "sigma": float(sigma), "outside_share": s0},
    )
