"""Common interface of the demand systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..market import Market
from ..units import FloatArray


class Demand(ABC):
    """A demand system for ``n_products`` differentiated products.

    Conventions: ``quantities(p)`` are unit sales; ``jacobian(p)[j, k]`` is the
    derivative of the quantity of product j with respect to the price of product k.
    Implementations must be analytic in ``p`` (they are evaluated at complex prices
    for complex-step derivatives), so no absolute values or comparisons on ``p``.
    """

    name: str = "demand"
    supports_zeta: bool = False

    @property
    @abstractmethod
    def n_products(self) -> int: ...

    @abstractmethod
    def quantities(self, p: FloatArray) -> FloatArray: ...

    @abstractmethod
    def jacobian(self, p: FloatArray) -> FloatArray: ...

    @abstractmethod
    def consumer_surplus(self, p: FloatArray) -> float:
        """Consumer surplus in currency units, defined up to a constant.

        Only differences between price vectors are meaningful.
        """

    def zeta_terms(self, p: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``(lam, gamma)`` with ``J = gamma - diag(lam)`` for Morrow-Skerlos iteration."""
        raise NotImplementedError(f"{self.name} has no zeta decomposition")

    def is_valid(self, p: FloatArray) -> bool:
        """Whether ``p`` lies in the region where the system is economically meaningful."""
        p = np.asarray(p)
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            return False
        q = self.quantities(p)
        return bool(np.all(np.isfinite(q)) and np.all(q > 0))

    def elasticities(self, p: FloatArray) -> FloatArray:
        """Price elasticities ``e[j, k] = d ln q_j / d ln p_k``."""
        q = self.quantities(p)
        return self.jacobian(p) * np.asarray(p)[None, :] / q[:, None]

    def diversion_quantity(self, p: FloatArray) -> FloatArray:
        """Quantity diversion ratios ``D[i, j] = -(dq_j/dp_i) / (dq_i/dp_i)``."""
        jac = self.jacobian(p)
        d = -jac.T / np.diag(jac)[:, None]
        np.fill_diagonal(d, 0.0)
        return d


@dataclass(frozen=True)
class Calibration:
    """A demand system calibrated to a market, with recovered costs and diagnostics."""

    demand: Demand
    market: Market
    costs: FloatArray
    fitted_margins: FloatArray
    margin_residuals: FloatArray
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.demand.name

    @property
    def max_margin_residual(self) -> float:
        r = self.margin_residuals[np.isfinite(self.margin_residuals)]
        return float(np.max(np.abs(r))) if r.size else 0.0
