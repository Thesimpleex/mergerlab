"""CES demand (revenue shares) with an outside good."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from ..errors import CalibrationError
from ..market import Market
from ..supply import recover_costs
from ..units import FloatArray, ShareBasis
from .base import Calibration, Demand


class CES(Demand):
    """CES demand with a numeraire outside good.

    Consumers with income ``Y`` spend revenue shares

        w_j = beta_j p_j^(1-sigma) / (1 + sum_k beta_k p_k^(1-sigma)),   q_j = Y w_j / p_j,

    and the remainder ``w_0 = 1 - sum_j w_j`` on the outside good, whose price is the
    numeraire. ``sigma > 1`` is the elasticity of substitution. The money-metric
    compensating variation of a move from ``p`` to ``p'`` is closed form,

        CV = Y [ (w_0(p') / w_0(p))^(1/(sigma-1)) - 1 ].
    """

    name = "CES"

    def __init__(
        self, beta: FloatArray, sigma: float, income: float, ref_prices: FloatArray
    ) -> None:
        if not sigma > 1:
            raise ValueError("the CES elasticity of substitution sigma must exceed 1")
        if not income > 0:
            raise ValueError("income must be positive")
        self.beta = np.asarray(beta, dtype=float)
        self.sigma = float(sigma)
        self.income = float(income)
        self.ref_prices = np.asarray(ref_prices, dtype=float)
        self._w0_ref = float(self.outside_share(self.ref_prices))

    @property
    def n_products(self) -> int:
        return self.beta.size

    def revenue_shares(self, p: FloatArray) -> FloatArray:
        t = self.beta * p ** (1.0 - self.sigma)
        return t / (1.0 + t.sum())

    def outside_share(self, p: FloatArray) -> float:
        return 1.0 / (1.0 + (self.beta * p ** (1.0 - self.sigma)).sum())

    def quantities(self, p: FloatArray) -> FloatArray:
        return self.income * self.revenue_shares(p) / p

    def jacobian(self, p: FloatArray) -> FloatArray:
        q = self.quantities(p)
        w = self.revenue_shares(p)
        eps = (self.sigma - 1.0) * np.ones((p.size, 1)) * w[None, :] - self.sigma * np.eye(p.size)
        return q[:, None] * eps / p[None, :]

    def consumer_surplus(self, p: FloatArray) -> float:
        """Money-metric welfare ``-Y (w_0(p)/w_0(p_ref))^(1/(sigma-1))``.

        Differences equal minus the compensating variation.
        """
        kappa = 1.0 / (self.sigma - 1.0)
        return float(-self.income * (self.outside_share(p) / self._w0_ref) ** kappa)


def _implied_margins(sigma: float, w: FloatArray, p: FloatArray, omega: FloatArray) -> FloatArray:
    """Lerner margins that the Bertrand conditions imply for CES revenue shares ``w``."""
    n = w.size
    w0 = 1.0 - w.sum()
    beta = w / (w0 * p ** (1.0 - sigma))
    model = CES(beta, sigma, 1.0, p)
    q = model.quantities(p)
    delta = omega * model.jacobian(p).T
    try:
        markup = -np.linalg.solve(delta, q)
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)
    return markup / p


def calibrate_ces(market: Market) -> Calibration:
    """CES with known outside share; ``sigma`` from the margins.

    Needs revenue shares of a market with an outside good and at least one Lerner
    margin. The Bertrand conditions make every margin a decreasing function of
    ``sigma`` (for one single-product firm ``m = 1 / (sigma - (sigma - 1) w)``); with
    several margins ``sigma`` is the least-squares fit on Lerner margins.
    """
    market.shares.require(ShareBasis.REVENUE, True, "CES")
    w = market.shares.values
    w0 = float(market.shares.outside)  # type: ignore[arg-type]
    p = market.prices
    omega = market.ownership.matrix
    m_obs = market.lerner
    known = np.isfinite(m_obs)
    if not known.any():
        raise CalibrationError("at least one margin is required to calibrate sigma")

    def resid(t: FloatArray) -> FloatArray:
        m = _implied_margins(1.0 + float(np.exp(t[0])), w, p, omega)
        r = (m - m_obs)[known]
        return np.where(np.isfinite(r), r, 1e3)

    sol = optimize.least_squares(
        resid, np.array([0.0]), bounds=([-14.0], [10.0]), xtol=1e-15, ftol=1e-15, gtol=1e-15
    )
    sigma = 1.0 + float(np.exp(sol.x[0]))
    if sol.x[0] <= -13.9 or sol.x[0] >= 9.9:
        raise CalibrationError(
            f"margins imply an elasticity of substitution at the search boundary ({sigma:.4g}); "
            "they are not compatible with CES demand and these shares"
        )
    beta = w / (w0 * p ** (1.0 - sigma))
    income = market.inside_revenue / (1.0 - w0)
    demand = CES(beta, sigma, income, p)
    costs = recover_costs(demand, p, omega)
    fitted = (p - costs) / p
    resid_full = np.full(m_obs.shape, np.nan)
    resid_full[known] = (fitted - m_obs)[known]
    return Calibration(
        demand, market, costs, fitted, resid_full, {"sigma": sigma, "outside_share": w0}
    )
