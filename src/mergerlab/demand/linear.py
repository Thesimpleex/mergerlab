"""Linear demand calibrated from margins and diversion ratios."""

from __future__ import annotations

import numpy as np

from ..errors import CalibrationError
from ..market import Market
from ..supply import recover_costs
from ..units import Diversion, FloatArray, ShareBasis
from .base import Calibration, Demand


class Linear(Demand):
    """``q = a + B p``.

    The Jacobian is the constant matrix ``B``. Consumer surplus is the line integral of
    ``-q . dp`` along the straight path from the reference prices; it is exact
    compensating variation only when ``B`` is symmetric (no income effects).
    """

    name = "linear"

    def __init__(self, intercept: FloatArray, slopes: FloatArray, ref_prices: FloatArray) -> None:
        self.intercept = np.asarray(intercept, dtype=float)
        self.slopes = np.asarray(slopes, dtype=float)
        self.ref_prices = np.asarray(ref_prices, dtype=float)
        if self.slopes.shape != (self.intercept.size,) * 2:
            raise ValueError("slopes must be a square matrix matching the intercept")

    @property
    def n_products(self) -> int:
        return self.intercept.size

    def quantities(self, p: FloatArray) -> FloatArray:
        return self.intercept + self.slopes @ p

    def jacobian(self, p: FloatArray) -> FloatArray:
        return self.slopes

    def consumer_surplus(self, p: FloatArray) -> float:
        delta = p - self.ref_prices
        q_ref = self.intercept + self.slopes @ self.ref_prices
        return float(-(q_ref @ delta + 0.5 * delta @ self.slopes @ delta))


def calibrate_linear(market: Market, diversion: Diversion | None = None) -> Calibration:
    """Linear demand from margins (all products) and quantity diversion ratios.

    For product j the first-order condition with diversion ``D_jk = -B_kj / B_jj``
    gives ``B_jj = -q_j / [a_j - sum_{k != j} Omega_jk D_jk a_k]`` with absolute
    margins ``a_k = m_k p_k``. Without ``diversion`` the ratios are proportional to
    quantity shares (``D_jk = s_k / (1 - s_j)``, shares including the outside good
    when the market has one).
    """
    p = market.prices
    omega = market.ownership.matrix
    m = market.lerner
    if not np.all(np.isfinite(m)):
        missing = np.flatnonzero(~np.isfinite(m)).tolist()
        raise CalibrationError(f"linear demand needs a margin for every product; missing {missing}")
    q = market.quantities()
    if diversion is None:
        if market.shares.has_outside:
            market.shares.require(ShareBasis.QUANTITY, True, "linear demand (default diversion)")
            s = market.shares.values
        else:
            s = market.quantity_shares_within()
        d = s[None, :] / (1.0 - s[:, None])
        np.fill_diagonal(d, 0.0)
    else:
        d = diversion.require_quantity("linear demand")
        if d.shape != (p.size, p.size):
            raise ValueError("diversion matrix must be J x J")
    a = m * p
    off = omega * d
    denom = a - off @ a
    if np.any(denom <= 0):
        raise CalibrationError(
            "diversion ratios and margins imply a non-negative own-price slope for products "
            f"{np.flatnonzero(denom <= 0).tolist()}"
        )
    b_diag = -q / denom
    slopes = -(d.T) * b_diag[None, :] + np.diag(b_diag)
    intercept = q - slopes @ p
    demand = Linear(intercept, slopes, p)
    costs = recover_costs(demand, p, omega)
    fitted = (p - costs) / p
    return Calibration(demand, market, costs, fitted, fitted - m, {})
