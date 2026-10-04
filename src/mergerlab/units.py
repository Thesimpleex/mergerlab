"""Typed inputs that make convention mixes impossible.

Three conventions are routinely confused in merger work and each of them changes
the answer:

* **Share basis.** Shares of units sold (quantity) or of sales value (revenue),
  and shares of the inside products only or of a market that includes an outside
  good.
* **Margin definition.** The Lerner index ``(p - c) / p`` or the absolute margin
  ``p - c``.
* **Diversion definition.** Units diverted (quantity), sales value diverted
  (value), the ratio of revenue changes (revenue) or the ratio of budget-share
  slopes reported by AIDS-type calibrations (share).

Each quantity is wrapped in a small immutable object that records its convention.
Conversions are explicit methods; functions that need a specific convention refuse
the others with a ``ValueError`` that names the conversion to call.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

_SUM_TOL = 1e-8


class ShareBasis(StrEnum):
    """What a share is a share of."""

    QUANTITY = "quantity"
    REVENUE = "revenue"


class MarginKind(StrEnum):
    """How a price-cost margin is expressed."""

    LERNER = "lerner"
    ABSOLUTE = "absolute"


class DiversionBasis(StrEnum):
    """How a diversion ratio is measured.

    ``QUANTITY``
        ``D_ij = -(dq_j/dp_i) / (dq_i/dp_i)``: units gained by j per unit lost by i.
    ``VALUE``
        ``D_ij * p_j / p_i``: sales value gained by j per unit of i's price-weighted
        lost sales (the "value diversion ratio" of competition authorities).
    ``REVENUE``
        ``-(dr_j/dp_i) / (dr_i/dp_i)`` with ``r = p q``: the ratio of *total* revenue
        changes, which includes the revenue effect of i's own price change.
    ``SHARE``
        ``-(dw_j/dp_i) / (dw_i/dp_i)`` with ``w`` the revenue shares within the market:
        the ratio of budget-share slopes. This is what R's ``antitrust`` reports for AIDS
        and PCAIDS (``-t(slopes) / diag(slopes)``). It coincides with ``REVENUE`` only
        when the market elasticity is -1; converting it needs the shares and the market
        elasticity as well.
    """

    QUANTITY = "quantity"
    VALUE = "value"
    REVENUE = "revenue"
    SHARE = "share"


def _vector(values: ArrayLike, name: str, allow_nan: bool = False) -> FloatArray:
    arr = np.array(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not allow_nan and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if allow_nan and np.any(np.isinf(arr)):
        raise ValueError(f"{name} must not contain infinite values")
    return arr


def check_prices(prices: ArrayLike) -> FloatArray:
    """Return prices as a float vector after checking that they are positive."""
    p = _vector(prices, "prices")
    if np.any(p <= 0):
        raise ValueError("prices must be strictly positive")
    return p


@dataclass(frozen=True)
class Shares:
    """Market shares with an explicit basis and outside-good convention.

    ``values`` are the shares of the inside products. If ``outside`` is ``None`` the
    values sum to one (shares *within* the market; no outside good). Otherwise the
    values are shares of a total market that also contains an outside good with
    share ``outside``, and they sum to ``1 - outside``.

    Use :meth:`within` and :meth:`total` to construct instances.
    """

    values: FloatArray
    basis: ShareBasis
    outside: float | None = None

    def __post_init__(self) -> None:
        v = _vector(self.values, "shares")
        if np.any(v <= 0):
            raise ValueError("shares must be strictly positive")
        total = float(v.sum())
        if self.outside is None:
            if abs(total - 1.0) > _SUM_TOL:
                raise ValueError(
                    f"shares within the market must sum to 1 (got {total:.6f}); "
                    "use Shares.total(...) if they are shares of a market that "
                    "includes an outside good"
                )
            v = v / total
        else:
            if not 0.0 < self.outside < 1.0:
                raise ValueError("outside share must lie strictly between 0 and 1")
            if abs(total + self.outside - 1.0) > _SUM_TOL:
                raise ValueError(
                    f"inside shares ({total:.6f}) plus outside share ({self.outside:.6f}) "
                    "must sum to 1"
                )
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "basis", ShareBasis(self.basis))

    @classmethod
    def within(cls, values: ArrayLike, basis: ShareBasis | str) -> Shares:
        """Shares of the inside products only; they must sum to one."""
        return cls(np.asarray(values, dtype=float), ShareBasis(basis), None)

    @classmethod
    def total(
        cls, values: ArrayLike, basis: ShareBasis | str, outside: float | None = None
    ) -> Shares:
        """Shares of a market that includes an outside good.

        If ``outside`` is omitted it is ``1 - sum(values)``, which must be positive.
        """
        v = np.asarray(values, dtype=float)
        s0 = 1.0 - float(v.sum()) if outside is None else float(outside)
        if outside is None and s0 <= _SUM_TOL:
            raise ValueError("shares sum to one: there is no room for an outside good")
        return cls(v, ShareBasis(basis), s0)

    @property
    def has_outside(self) -> bool:
        return self.outside is not None

    def within_values(self) -> FloatArray:
        """Inside shares renormalised to sum to one."""
        return self.values / self.values.sum()

    def require(self, basis: ShareBasis, outside: bool | None, who: str) -> None:
        """Raise ``ValueError`` unless the shares have the convention ``who`` needs."""
        if self.basis is not basis:
            other = "quantity" if basis is ShareBasis.REVENUE else "revenue"
            raise ValueError(
                f"{who} requires {basis.value} shares but received {other} shares; "
                "convert inside-market shares with to_quantity()/to_revenue(prices) first"
            )
        if outside is True and not self.has_outside:
            raise ValueError(
                f"{who} requires shares of a market with an outside good "
                "(construct them with Shares.total)"
            )
        if outside is False and self.has_outside:
            raise ValueError(
                f"{who} requires shares within the market (no outside good); "
                "use Shares.within or renormalise"
            )

    def _convert(
        self, prices: ArrayLike, target: ShareBasis, outside_price: float | None
    ) -> Shares:
        if self.basis is target:
            return self
        p = check_prices(prices)
        if p.shape != self.values.shape:
            raise ValueError("prices and shares must have the same length")
        scale = p if target is ShareBasis.REVENUE else 1.0 / p
        if not self.has_outside:
            scaled = self.values * scale
            return Shares.within(scaled / scaled.sum(), target)
        if outside_price is None or not outside_price > 0:
            raise ValueError(
                "shares that include an outside good cannot be converted between quantity "
                "and revenue bases without the outside good's price: pass outside_price (set "
                "outside_price in [market] of a case file), or convert the inside shares with "
                "Shares.within and add the outside share on the new basis with with_outside()"
            )
        out_scale = outside_price if target is ShareBasis.REVENUE else 1.0 / outside_price
        inside = self.values * scale
        outside = float(self.outside) * out_scale  # type: ignore[arg-type]
        total = inside.sum() + outside
        return Shares(inside / total, target, outside / total)

    def to_revenue(self, prices: ArrayLike, outside_price: float | None = None) -> Shares:
        """Quantity shares to revenue shares: ``w_j`` proportional to ``p_j s_j``.

        With an outside good its price ``outside_price`` is required.
        """
        return self._convert(prices, ShareBasis.REVENUE, outside_price)

    def to_quantity(self, prices: ArrayLike, outside_price: float | None = None) -> Shares:
        """Revenue shares to quantity shares: ``s_j`` proportional to ``w_j / p_j``.

        With an outside good its price ``outside_price`` is required.
        """
        return self._convert(prices, ShareBasis.QUANTITY, outside_price)

    def with_outside(self, outside: float) -> Shares:
        """Rescale within-market shares so that an outside good has share ``outside``."""
        if self.has_outside:
            raise ValueError("shares already include an outside good")
        if not 0.0 < outside < 1.0:
            raise ValueError("outside share must lie strictly between 0 and 1")
        return Shares(self.values * (1.0 - outside), self.basis, float(outside))


@dataclass(frozen=True)
class Margins:
    """Price-cost margins with an explicit definition.

    Unknown margins are ``nan``; at least one margin is needed by most calibrations.
    """

    values: FloatArray
    kind: MarginKind

    def __post_init__(self) -> None:
        v = _vector(self.values, "margins", allow_nan=True)
        known = np.isfinite(v)
        if np.any(v[known] <= 0):
            raise ValueError("margins must be strictly positive")
        if self.kind is MarginKind.LERNER and np.any(v[known] >= 1):
            raise ValueError(
                "Lerner margins (p - c) / p must lie in (0, 1); if these are absolute "
                "margins p - c use Margins.absolute(...)"
            )
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "kind", MarginKind(self.kind))

    @classmethod
    def lerner(cls, values: ArrayLike) -> Margins:
        """Margins as a fraction of price, ``(p - c) / p``."""
        return cls(np.asarray(values, dtype=float), MarginKind.LERNER)

    @classmethod
    def absolute(cls, values: ArrayLike) -> Margins:
        """Margins in currency per unit, ``p - c``."""
        return cls(np.asarray(values, dtype=float), MarginKind.ABSOLUTE)

    @property
    def known(self) -> NDArray[np.bool_]:
        return np.isfinite(self.values)

    def as_lerner(self, prices: ArrayLike) -> FloatArray:
        """Lerner margins (``nan`` where unknown)."""
        p = check_prices(prices)
        if p.shape != self.values.shape:
            raise ValueError("prices and margins must have the same length")
        if self.kind is MarginKind.LERNER:
            return self.values.copy()
        m = self.values / p
        if np.any(m[self.known] >= 1):
            raise ValueError("absolute margins must be smaller than prices")
        return m

    def as_absolute(self, prices: ArrayLike) -> FloatArray:
        """Absolute margins ``p - c`` (``nan`` where unknown)."""
        p = check_prices(prices)
        if p.shape != self.values.shape:
            raise ValueError("prices and margins must have the same length")
        if self.kind is MarginKind.ABSOLUTE:
            if np.any(self.values[self.known] >= p[self.known]):
                raise ValueError("absolute margins must be smaller than prices")
            return self.values.copy()
        return self.values * p


@dataclass(frozen=True)
class Diversion:
    """A diversion-ratio matrix with an explicit definition.

    ``matrix[i, j]`` is the diversion from product i to product j; the diagonal is
    ignored and stored as zero.

    Quantity diversions supplied by a user (:meth:`quantity`) must sum to at most one
    over the rivals of each product: more units cannot be diverted than were lost. That
    bound holds only when units are commensurable. A demand system at unequal prices
    (CES, PCAIDS, linear) can imply row sums above one because spending moves to cheaper
    goods, so matrices derived from a demand system or from a conversion are built with
    :meth:`from_demand`, which does not apply the bound.
    """

    matrix: FloatArray
    basis: DiversionBasis
    bounded: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("diversion matrix must be square")
        if not np.all(np.isfinite(m)):
            raise ValueError("diversion matrix must be finite")
        np.fill_diagonal(m, 0.0)
        if np.any(m < 0):
            raise ValueError("diversion ratios must be non-negative")
        basis = DiversionBasis(self.basis)
        if self.bounded and basis is DiversionBasis.QUANTITY and np.any(m.sum(axis=1) > 1 + 1e-9):
            raise ValueError(
                "quantity diversion ratios out of a product cannot sum to more than 1; "
                "if these are revenue, value or share diversions declare that basis"
            )
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "basis", basis)

    @classmethod
    def quantity(cls, matrix: ArrayLike) -> Diversion:
        """User-supplied quantity diversions (rows sum to at most one)."""
        return cls(np.asarray(matrix, dtype=float), DiversionBasis.QUANTITY)

    @classmethod
    def from_demand(cls, matrix: ArrayLike) -> Diversion:
        """Quantity diversions implied by a demand system, without the row-sum bound."""
        return cls(np.asarray(matrix, dtype=float), DiversionBasis.QUANTITY, bounded=False)

    @classmethod
    def value(cls, matrix: ArrayLike) -> Diversion:
        return cls(np.asarray(matrix, dtype=float), DiversionBasis.VALUE)

    @classmethod
    def revenue(cls, matrix: ArrayLike) -> Diversion:
        return cls(np.asarray(matrix, dtype=float), DiversionBasis.REVENUE)

    @classmethod
    def share(cls, matrix: ArrayLike) -> Diversion:
        """Budget-share-slope diversions, the AIDS convention of R's ``antitrust``."""
        return cls(np.asarray(matrix, dtype=float), DiversionBasis.SHARE)

    def to_quantity(
        self,
        prices: ArrayLike,
        own_elasticity: ArrayLike | None = None,
        *,
        shares: ArrayLike | None = None,
        market_elasticity: float | None = None,
    ) -> Diversion:
        """Convert to the unit-based diversion ratio used by GUPPI, UPP and CMCR.

        ``own_elasticity`` is needed for the revenue and share bases; the share basis
        also needs the within-market revenue ``shares`` and the ``market_elasticity``.
        """
        if self.basis is DiversionBasis.QUANTITY:
            return self
        p = check_prices(prices)
        ratio = p[None, :] / p[:, None]
        if self.basis is DiversionBasis.VALUE:
            return Diversion.from_demand(self.matrix / ratio)
        if self.basis is DiversionBasis.SHARE:
            e, w, k = self._share_terms(p, own_elasticity, shares, market_elasticity)
            denom = e + 1.0 - k * w
            numer = self.matrix * denom[:, None] - k * w[None, :]
            return Diversion.from_demand(numer / e[:, None] / ratio)
        e = self._own_elasticity(p, own_elasticity)
        return Diversion.from_demand(self.matrix / ratio * ((1.0 + e) / e)[:, None])

    def to_value(
        self,
        prices: ArrayLike,
        own_elasticity: ArrayLike | None = None,
        *,
        shares: ArrayLike | None = None,
        market_elasticity: float | None = None,
    ) -> Diversion:
        """Convert to the sales-value diversion ratio ``D_ij p_j / p_i``."""
        if self.basis is DiversionBasis.VALUE:
            return self
        p = check_prices(prices)
        ratio = p[None, :] / p[:, None]
        q = self.to_quantity(p, own_elasticity, shares=shares, market_elasticity=market_elasticity)
        return Diversion.value(q.matrix * ratio)

    def to_revenue(
        self,
        prices: ArrayLike,
        own_elasticity: ArrayLike,
        *,
        shares: ArrayLike | None = None,
        market_elasticity: float | None = None,
    ) -> Diversion:
        """Convert to the revenue-change ratio ``-(dr_j/dp_i) / (dr_i/dp_i)``."""
        if self.basis is DiversionBasis.REVENUE:
            return self
        p = check_prices(prices)
        e = self._own_elasticity(p, own_elasticity)
        ratio = p[None, :] / p[:, None]
        q = self.to_quantity(p, e, shares=shares, market_elasticity=market_elasticity)
        return Diversion.revenue(q.matrix * ratio * (e / (1.0 + e))[:, None])

    def to_share(
        self,
        prices: ArrayLike,
        own_elasticity: ArrayLike,
        shares: ArrayLike,
        market_elasticity: float,
    ) -> Diversion:
        """Convert to the budget-share-slope ratio ``-(dw_j/dp_i) / (dw_i/dp_i)`` of AIDS.

        Under the PCAIDS expenditure equation, with ``k = eps_m + 1`` and ``e_i`` the own
        elasticity, ``D_ij = (p_i / p_j) [S_ij (e_i + 1 - k w_i) - k w_j] / e_i``; this
        inverts it.
        """
        if self.basis is DiversionBasis.SHARE:
            return self
        p = check_prices(prices)
        e, w, k = self._share_terms(p, own_elasticity, shares, market_elasticity)
        ratio = p[None, :] / p[:, None]
        q = self.to_quantity(p, e, shares=w, market_elasticity=market_elasticity)
        numer = q.matrix * ratio * e[:, None] + k * w[None, :]
        return Diversion.share(numer / (e + 1.0 - k * w)[:, None])

    @staticmethod
    def _share_terms(
        prices: FloatArray,
        own_elasticity: ArrayLike | None,
        shares: ArrayLike | None,
        market_elasticity: float | None,
    ) -> tuple[FloatArray, FloatArray, float]:
        if own_elasticity is None or shares is None or market_elasticity is None:
            raise ValueError(
                "converting to or from a share (AIDS) diversion needs own_elasticity, the "
                "within-market revenue shares and the market_elasticity: the share-slope "
                "ratio equals the revenue ratio only when the market elasticity is -1"
            )
        n = prices.size
        e = _vector(own_elasticity, "own_elasticity")
        w = _vector(shares, "shares")
        if e.size != n or w.size != n:
            raise ValueError("own_elasticity and shares need one entry per product")
        if np.any(e >= 0):
            raise ValueError("own_elasticity must be negative")
        if np.any(w <= 0) or abs(float(w.sum()) - 1.0) > 1e-6:
            raise ValueError("shares must be positive within-market revenue shares summing to 1")
        if not market_elasticity < 0:
            raise ValueError("market_elasticity must be negative")
        k = float(market_elasticity) + 1.0
        if np.any(e + 1.0 - k * w >= 0):
            raise ValueError(
                "the own elasticities are not compatible with the market elasticity and "
                "shares (an own share slope would be non-negative)"
            )
        return e, w, k

    @staticmethod
    def _own_elasticity(prices: FloatArray, own_elasticity: ArrayLike | None) -> FloatArray:
        if own_elasticity is None:
            raise ValueError(
                "converting to or from a revenue diversion needs the own-price "
                "elasticities (own_elasticity): R_ij = D_ij (p_j / p_i) e_ii / (1 + e_ii)"
            )
        e = _vector(own_elasticity, "own_elasticity")
        if e.size != prices.size or np.any(e >= -1.0):
            raise ValueError("own_elasticity must have one entry per product, all below -1")
        return e

    def require_quantity(self, who: str) -> FloatArray:
        """Return the matrix, or raise unless the diversion is quantity-based."""
        if self.basis is not DiversionBasis.QUANTITY:
            raise ValueError(
                f"{who} requires quantity diversion ratios but received "
                f"{self.basis.value} diversions; convert with Diversion.to_quantity(prices, "
                "own_elasticity) first (share diversions also need shares and market_elasticity)"
            )
        return self.matrix


@dataclass(frozen=True)
class Ownership:
    """Ownership matrix ``Omega``: entry (j, k) is 1 if j and k are priced jointly.

    Fractional entries encode partial ownership or common-ownership weights in the
    sense of the first-order condition ``q + (Omega * J') (p - c) = 0``.
    """

    matrix: FloatArray
    owners: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("ownership matrix must be square")
        if not np.all(np.isfinite(m)) or np.any(m < 0) or np.any(m > 1):
            raise ValueError("ownership entries must lie in [0, 1]")
        if not np.allclose(np.diag(m), 1.0):
            raise ValueError("the diagonal of an ownership matrix must be 1")
        object.__setattr__(self, "matrix", m)

    @classmethod
    def from_owners(cls, owners: Sequence[str]) -> Ownership:
        """Block ownership: products with the same owner label are priced jointly."""
        labels = tuple(str(o) for o in owners)
        arr = np.array(labels)
        return cls((arr[:, None] == arr[None, :]).astype(float), labels)

    @property
    def n_products(self) -> int:
        return self.matrix.shape[0]

    @property
    def is_binary(self) -> bool:
        return bool(np.all((self.matrix == 0) | (self.matrix == 1)))

    def merged(self, parties: Iterable[str]) -> Ownership:
        """Ownership after the listed owners combine (requires owner labels)."""
        if self.owners is None:
            raise ValueError("merged() needs an ownership built with from_owners")
        parties = tuple(parties)
        missing = set(parties) - set(self.owners)
        if missing:
            raise ValueError(f"unknown owner(s): {sorted(missing)}")
        if len(set(parties)) < 2:
            raise ValueError("a merger needs at least two distinct owners")
        new = tuple(parties[0] if o in parties else o for o in self.owners)
        return Ownership.from_owners(new)

    def firm_shares(self, shares: ArrayLike) -> FloatArray:
        """Share of the firm that owns each product (binary ownership only)."""
        if not self.is_binary:
            raise ValueError("firm shares need a binary ownership matrix")
        s = _vector(shares, "shares")
        return self.matrix @ s

    def merging_products(self, post: Ownership) -> NDArray[np.bool_]:
        """Products whose ownership row changes between this and ``post``."""
        if post.matrix.shape != self.matrix.shape:
            raise ValueError("ownership matrices must have the same shape")
        return np.any(np.abs(post.matrix - self.matrix) > 1e-12, axis=1)
