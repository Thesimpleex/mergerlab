"""Proportionality-calibrated AIDS (Epstein and Rubinfeld, 2002)."""

from __future__ import annotations

import numpy as np

from ..errors import CalibrationError
from ..market import Market
from ..supply import recover_costs
from ..units import FloatArray, ShareBasis
from .base import Calibration, Demand

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)


class PCAIDS(Demand):
    """AIDS with unit income elasticity whose slopes are calibrated by proportionality.

    Revenue shares within the market are linear in log prices,

        w(p) = w_ref + B ln(p / p_ref),   B = -k (diag(w_ref) - w_ref w_ref'),

    so the revenue diversion from i to j is ``w_j / (1 - w_i)`` (diversion proportional
    to share, the PCAIDS restriction). The market (group) expenditure ``Y_g`` responds to
    the Divisia price index with elasticity ``eps_m + 1``:

        ln(Y_g / Y_ref) = (eps_m + 1) (w_ref . x + x' B x / 2),   x = ln(p / p_ref),

    and ``q_i = Y_g w_i / p_i``. Elasticities are
    ``e_ij = B_ij / w_i + (eps_m + 1) w_j - delta_ij``.
    """

    name = "PCAIDS"

    def __init__(
        self,
        w_ref: FloatArray,
        slopes: FloatArray,
        market_elasticity: float,
        expenditure: float,
        ref_prices: FloatArray,
    ) -> None:
        self.w_ref = np.asarray(w_ref, dtype=float)
        self.slopes = np.asarray(slopes, dtype=float)
        self.market_elasticity = float(market_elasticity)
        self.expenditure = float(expenditure)
        self.ref_prices = np.asarray(ref_prices, dtype=float)

    @property
    def n_products(self) -> int:
        return self.w_ref.size

    def revenue_shares(self, p: FloatArray) -> FloatArray:
        x = np.log(p / self.ref_prices)
        return self.w_ref + self.slopes @ x

    def group_expenditure(self, p: FloatArray) -> float:
        x = np.log(p / self.ref_prices)
        index = self.w_ref @ x + 0.5 * x @ (self.slopes @ x)
        return self.expenditure * np.exp((self.market_elasticity + 1.0) * index)

    def quantities(self, p: FloatArray) -> FloatArray:
        return self.group_expenditure(p) * self.revenue_shares(p) / p

    def jacobian(self, p: FloatArray) -> FloatArray:
        w = self.revenue_shares(p)
        q = self.quantities(p)
        eps = (
            self.slopes / w[:, None]
            + (self.market_elasticity + 1.0) * np.ones((p.size, 1)) * w[None, :]
            - np.eye(p.size)
        )
        return q[:, None] * eps / p[None, :]

    def is_valid(self, p: FloatArray) -> bool:
        p = np.asarray(p)
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            return False
        w = self.revenue_shares(p)
        return bool(np.all(np.isfinite(w)) and np.all(w > 0))

    def consumer_surplus(self, p: FloatArray) -> float:
        """Marshallian surplus ``-integral q . dp`` along the straight path from ``p_ref``.

        With ``eps_m = -1`` the integrand is a gradient and the value equals minus group
        expenditure times the change in the log price index.
        """
        delta = p - self.ref_prices
        total = 0.0
        for t, w in zip(0.5 * (_GL_NODES + 1.0), 0.5 * _GL_WEIGHTS, strict=True):
            pt = self.ref_prices + t * delta
            total += w * float(self.quantities(pt) @ delta)
        return -total


def calibrate_pcaids(
    market: Market,
    own_elasticity: float,
    market_elasticity: float = -1.0,
    known_index: int = 0,
) -> Calibration:
    """PCAIDS from within-market revenue shares, the market elasticity and one own elasticity.

    The own-price elasticity ``e`` of product ``known_index`` fixes the slope
    ``B_ii = w_i (e + 1 - w_i (eps_m + 1))``; all other slopes follow from proportional
    diversion. Margins are implied by the elasticities (and need not be supplied). If the
    market carries margins they are reported against the implied ones as residuals but
    are not used.
    """
    market.shares.require(ShareBasis.REVENUE, False, "PCAIDS")
    w = market.shares.values
    p = market.prices
    omega = market.ownership.matrix
    n = w.size
    if not 0 <= known_index < n:
        raise ValueError("known_index out of range")
    if not own_elasticity < -1.0:
        raise ValueError("own_elasticity must be below -1")
    if not market_elasticity < 0.0:
        raise ValueError("market_elasticity must be negative")
    wi = w[known_index]
    b_known = wi * (own_elasticity + 1.0 - wi * (market_elasticity + 1.0))
    if not b_known < 0:
        raise CalibrationError(
            "the own-price elasticity is not compatible with the market elasticity and "
            "share (the implied own share slope is non-negative)"
        )
    k = -b_known / (wi * (1.0 - wi))
    slopes = -k * (np.diag(w) - np.outer(w, w))
    expenditure = market.inside_revenue
    demand = PCAIDS(w, slopes, market_elasticity, expenditure, p)
    costs = recover_costs(demand, p, omega)
    fitted = (p - costs) / p
    m_obs = market.lerner
    resid = np.where(np.isfinite(m_obs), fitted - m_obs, np.nan)
    return Calibration(
        demand,
        market,
        costs,
        fitted,
        resid,
        {"own_elasticity": float(own_elasticity), "market_elasticity": float(market_elasticity)},
    )
