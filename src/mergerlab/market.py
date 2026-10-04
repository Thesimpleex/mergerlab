"""Observed pre-merger market data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .units import FloatArray, Margins, Ownership, ShareBasis, Shares, check_prices


@dataclass(frozen=True)
class Market:
    """Pre-merger prices, shares, margins and ownership of the products in a market.

    Parameters
    ----------
    prices
        Pre-merger prices (strictly positive, one currency unit per unit sold).
    shares
        Typed shares; see :class:`mergerlab.units.Shares`.
    margins
        Typed margins with ``nan`` for unknown entries.
    ownership
        Pre-merger ownership.
    labels
        Product names.
    revenue
        Optional total sales revenue of the inside products at pre-merger prices, in
        currency units. It fixes the absolute scale (units sold, dollar synergies,
        consumer surplus in currency). If omitted everything is per unit of inside
        sales (inside quantity normalised to one).
    outside_price
        Price of the outside good. Needed only to translate shares that include an
        outside good between quantity and revenue bases (:meth:`as_basis`).
    """

    prices: FloatArray
    shares: Shares
    margins: Margins
    ownership: Ownership
    labels: tuple[str, ...]
    revenue: float | None = None
    outside_price: float | None = None

    def __post_init__(self) -> None:
        p = check_prices(self.prices)
        n = p.size
        if self.shares.values.size != n:
            raise ValueError("shares and prices must have the same length")
        if self.margins.values.size != n:
            raise ValueError("margins and prices must have the same length")
        if self.ownership.n_products != n:
            raise ValueError("ownership and prices must have the same length")
        if len(self.labels) != n:
            raise ValueError("labels and prices must have the same length")
        if self.revenue is not None and not self.revenue > 0:
            raise ValueError("revenue must be positive")
        if self.outside_price is not None and not self.outside_price > 0:
            raise ValueError("outside_price must be positive")
        object.__setattr__(self, "prices", p)
        object.__setattr__(self, "labels", tuple(self.labels))

    @classmethod
    def build(
        cls,
        prices: Sequence[float],
        shares: Shares,
        margins: Margins,
        owners: Sequence[str],
        labels: Sequence[str] | None = None,
        revenue: float | None = None,
        outside_price: float | None = None,
    ) -> Market:
        n = len(prices)
        names = tuple(labels) if labels is not None else tuple(f"P{i + 1}" for i in range(n))
        return cls(
            np.asarray(prices, dtype=float),
            shares,
            margins,
            Ownership.from_owners(owners),
            names,
            revenue,
            outside_price,
        )

    @property
    def n_products(self) -> int:
        return self.prices.size

    @property
    def lerner(self) -> FloatArray:
        """Lerner margins, ``nan`` where unknown."""
        return self.margins.as_lerner(self.prices)

    def quantity_shares_within(self) -> FloatArray:
        """Quantity shares of the inside products, summing to one."""
        v = self.shares.within_values()
        if self.shares.basis is ShareBasis.QUANTITY:
            return v
        v = v / self.prices
        return v / v.sum()

    def revenue_shares_within(self) -> FloatArray:
        """Revenue shares of the inside products, summing to one."""
        v = self.shares.within_values()
        if self.shares.basis is ShareBasis.REVENUE:
            return v
        v = v * self.prices
        return v / v.sum()

    def as_basis(self, basis: ShareBasis) -> Market:
        """The same market with shares expressed on ``basis``.

        Shares that include an outside good need ``outside_price`` for the conversion.
        """
        shares = (
            self.shares.to_quantity(self.prices, self.outside_price)
            if basis is ShareBasis.QUANTITY
            else self.shares.to_revenue(self.prices, self.outside_price)
        )
        return replace(self, shares=shares)

    def within(self) -> Market:
        """The same market with shares restricted to the inside products (sum to one)."""
        if not self.shares.has_outside:
            return self
        return replace(self, shares=Shares.within(self.shares.within_values(), self.shares.basis))

    def quantities(self) -> FloatArray:
        """Units sold of each inside product at pre-merger prices."""
        s = self.shares
        if s.basis is ShareBasis.QUANTITY:
            qs = s.within_values()
            total = 1.0 if self.revenue is None else self.revenue / float(self.prices @ qs)
            return total * qs
        rs = s.within_values()
        scale = 1.0 if self.revenue is None else self.revenue
        return scale * rs / self.prices

    @property
    def inside_revenue(self) -> float:
        """Total sales value of the inside products at pre-merger prices."""
        return float(self.prices @ self.quantities())
