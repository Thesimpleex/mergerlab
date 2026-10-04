"""Multinomial logit demand with an outside good, and its calibrations."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..errors import CalibrationError
from ..market import Market
from ..supply import recover_costs
from ..units import FloatArray, ShareBasis
from .base import Calibration, Demand

OUTSIDE_BOUNDS = (1e-3, 1.0 - 1e-3)


class Logit(Demand):
    """``s_j = exp(delta_j - alpha p_j) / (1 + sum_k exp(delta_k - alpha p_k))``.

    Unit sales are ``q_j = M s_j`` where ``M`` is the total market size including the
    outside good (utility normalised to zero).
    """

    name = "logit"
    supports_zeta = True

    def __init__(self, delta: FloatArray, alpha: float, market_size: float = 1.0) -> None:
        if not alpha > 0:
            raise ValueError("alpha must be positive")
        if not market_size > 0:
            raise ValueError("market_size must be positive")
        self.delta = np.asarray(delta, dtype=float)
        self.alpha = float(alpha)
        self.market_size = float(market_size)

    @property
    def n_products(self) -> int:
        return self.delta.size

    def shares(self, p: FloatArray) -> FloatArray:
        e = np.exp(self.delta - self.alpha * p)
        return e / (1.0 + e.sum())

    def quantities(self, p: FloatArray) -> FloatArray:
        return self.market_size * self.shares(p)

    def jacobian(self, p: FloatArray) -> FloatArray:
        s = self.shares(p)
        return self.market_size * self.alpha * (np.outer(s, s) - np.diag(s))

    def zeta_terms(self, p: FloatArray) -> tuple[FloatArray, FloatArray]:
        s = self.shares(p)
        m = self.market_size * self.alpha
        return m * s, m * np.outer(s, s)

    def consumer_surplus(self, p: FloatArray) -> float:
        """Log-sum surplus ``(M / alpha) ln(1 + sum_j exp(delta_j - alpha p_j))``."""
        return float(
            self.market_size / self.alpha * np.log1p(np.exp(self.delta - self.alpha * p).sum())
        )


def _markup_multipliers(shares: FloatArray, omega: FloatArray) -> FloatArray:
    """``g = (I - Omega diag(s))^{-1} 1`` so that absolute markups equal ``g / alpha``."""
    n = shares.size
    try:
        return np.asarray(
            np.linalg.solve(np.eye(n) - omega * shares[None, :], np.ones(n)), dtype=np.float64
        )
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("singular multi-product markup system") from exc


def _fit_inverse_scale(basis_vec: FloatArray, margins: FloatArray) -> tuple[float, FloatArray]:
    """Least squares for ``1 / alpha`` in ``m_j = (g_j / p_j) / alpha`` over known margins."""
    known = np.isfinite(margins)
    if not known.any():
        raise CalibrationError("at least one margin is required to calibrate the price coefficient")
    x = basis_vec[known]
    inv_alpha = float(x @ margins[known] / (x @ x))
    resid = np.full(margins.shape, np.nan)
    resid[known] = inv_alpha * x - margins[known]
    if not inv_alpha > 0:
        raise CalibrationError("margins imply a non-positive price coefficient")
    return inv_alpha, resid


def calibrate_logit(market: Market) -> Calibration:
    """Logit with known outside share; ``alpha`` from the multi-product markup condition.

    Needs quantity shares of a market that includes an outside good, and at least one
    Lerner margin. For single-product firms ``p_j - c_j = 1 / (alpha (1 - s_j))``; for
    a multi-product firm the common-markup condition holds with the firm's total share.
    With more than one margin ``1 / alpha`` is the least-squares fit on Lerner margins,
    and the residuals are reported (not hidden).
    """
    market.shares.require(ShareBasis.QUANTITY, True, "logit")
    s = market.shares.values
    s0 = float(market.shares.outside)  # type: ignore[arg-type]
    omega = market.ownership.matrix
    p = market.prices
    g = _markup_multipliers(s, omega)
    inv_alpha, resid = _fit_inverse_scale(g / p, market.lerner)
    alpha = 1.0 / inv_alpha
    delta = np.log(s / s0) + alpha * p
    q_inside = market.quantities()
    demand = Logit(delta, alpha, market_size=float(q_inside.sum()) / (1.0 - s0))
    costs = recover_costs(demand, p, omega)
    fitted = (p - costs) / p
    return Calibration(demand, market, costs, fitted, resid, {"alpha": alpha, "outside_share": s0})


def calibrate_logit_alm(
    market: Market, outside_bounds: tuple[float, float] = OUTSIDE_BOUNDS
) -> Calibration:
    """Logit with unknown outside share, calibrated from two or more margins ("ALM").

    The inside quantity shares ``s~`` are known and the outside share ``s0`` is a
    parameter: shares of the total market are ``(1 - s0) s~``. The multi-product markup
    condition then identifies ``(alpha, s0)`` from margins that vary with firm share.
    With exactly two distinguishable margins the fit is exact; with more it is least
    squares on Lerner margins.

    Raises
    ------
    CalibrationError
        If fewer than two margins are known, if margins do not vary with firm share,
        or if the implied outside share lies outside ``outside_bounds`` (near 0 or near
        1 the model is not identified and a silent answer would be arbitrary).
    """
    market.shares.require(ShareBasis.QUANTITY, False, "logit ALM")
    s_in = market.shares.values
    omega = market.ownership.matrix
    p = market.prices
    m = market.lerner
    known = np.isfinite(m)
    if known.sum() < 2:
        raise CalibrationError("logit ALM needs at least two margins to identify the outside share")
    firm_share = omega @ s_in
    if np.ptp(firm_share[known]) < 1e-9:
        raise CalibrationError(
            "the known margins belong to products whose firms have identical shares; "
            "the outside share is not identified"
        )

    def margins_for(alpha: float, b: float) -> FloatArray:
        return _markup_multipliers(b * s_in, omega) / (alpha * p)

    def residuals(theta: FloatArray) -> FloatArray:
        alpha, b = float(np.exp(theta[0])), float(theta[1])
        return (margins_for(alpha, b) - m)[known]

    a0 = 1.0 / (np.nanmean(m[known] * p[known]) * (1.0 - float(firm_share[known].mean()) * 0.5))
    best = None
    for b0 in (0.5, 0.2, 0.8, 0.05, 0.95):
        sol = optimize.least_squares(
            residuals,
            np.array([np.log(a0), b0]),
            bounds=([-30.0, 1e-9], [30.0, 1.0 - 1e-9]),
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    alpha, b = float(np.exp(best.x[0])), float(best.x[1])
    s0 = 1.0 - b
    lo, hi = outside_bounds
    if not lo <= s0 <= hi:
        side = "close to 0" if s0 < lo else "close to 1"
        raise CalibrationError(
            f"calibrated outside share is {s0:.4f}, {side} (admissible range {lo}-{hi}); "
            "the margins carry no usable information about the outside good here. "
            "Provide the outside share (outside_share in the case file, or Shares.total) instead"
        )
    resid = np.full(m.shape, np.nan)
    resid[known] = residuals(best.x)
    shares_total = b * s_in
    delta = np.log(shares_total / s0) + alpha * p
    q_inside = market.quantities()
    demand = Logit(delta, alpha, market_size=float(q_inside.sum()) / b)
    costs = recover_costs(demand, p, omega)
    fitted = (p - costs) / p
    return Calibration(demand, market, costs, fitted, resid, {"alpha": alpha, "outside_share": s0})
