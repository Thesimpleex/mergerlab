"""Monte Carlo bands over unobserved inputs and an ensemble over demand forms.

Margins, nesting parameters, market elasticities and diversion ratios are rarely observed
in a first-day screen. Each draw perturbs them within stated ranges, recalibrates every
demand system, solves the post-merger equilibrium under the solver gate and records the
price effect. Draws that cannot be calibrated or whose equilibrium fails the gate are
counted by cause and reported, and excluded from the bands. The bands are therefore
conditional on inputs that can be calibrated; :meth:`MonteCarloResult.covered_range` gives
the range of a drawn parameter that the successful draws actually cover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .errors import CalibrationError, EquilibriumNotFound
from .market import Market
from .metrics import cmcr_from_demand
from .simulate import ModelSpec, calibrate
from .supply import DEFAULT_GATE, solve_bertrand
from .units import Diversion, FloatArray, Margins, Ownership, ShareBasis

STATUSES = ("ok", "calibration_failed", "solver_failed")


@dataclass(frozen=True)
class Priors:
    """Ranges for the unobserved inputs.

    ``margin_halfwidth``
        Each supplied Lerner margin is drawn uniformly within plus or minus this absolute
        amount (clipped to 0.02 to 0.95); independently across products.
    ``sigma_range``
        Uniform range for the nesting parameter of nested-logit specifications.
    ``market_elasticity_range``
        Uniform range for the market elasticity of PCAIDS specifications.
    ``diversion_concentration``
        If set, the linear model's diversion ratios are drawn row by row from a Dirichlet
        distribution centred on the share-proportional default with this concentration
        (larger means tighter). ``None`` keeps the default diversions.
    """

    margin_halfwidth: float = 0.10
    sigma_range: tuple[float, float] | None = None
    market_elasticity_range: tuple[float, float] | None = None
    diversion_concentration: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.margin_halfwidth < 0.5:
            raise ValueError("margin_halfwidth must lie in [0, 0.5)")
        for name in ("sigma_range", "market_elasticity_range"):
            rng = getattr(self, name)
            if rng is not None and not rng[0] <= rng[1]:
                raise ValueError(f"{name} must be (low, high) with low <= high")
        if self.sigma_range is not None and not (
            self.sigma_range[0] >= 0 and self.sigma_range[1] < 1
        ):
            raise ValueError("sigma_range must lie within [0, 1)")
        if self.market_elasticity_range is not None and self.market_elasticity_range[1] >= 0:
            raise ValueError("market_elasticity_range must be negative")
        if self.diversion_concentration is not None and not self.diversion_concentration > 0:
            raise ValueError("diversion_concentration must be positive")


@dataclass
class MonteCarloResult:
    """Draws of price effects by model, with failure accounting."""

    specs: tuple[ModelSpec, ...]
    parties: tuple[int, ...]
    weights: FloatArray
    price_change: dict[str, FloatArray]
    party_change: dict[str, FloatArray]
    party_cmcr: dict[str, FloatArray]
    status: dict[str, list[str]]
    draws: int
    seed: int | None = None
    priors: Priors = field(default_factory=Priors)
    parameter_draws: dict[str, FloatArray] = field(default_factory=dict)

    def covered_range(
        self, kind: str, parameter: str, quantiles: tuple[float, float] = (0.0, 1.0)
    ) -> tuple[float, float] | None:
        """Range of a drawn parameter (``sigma`` or ``market_elasticity``) among successful draws.

        Failures concentrate at extreme parameter values, so this can be much narrower than
        the prior range the draws were taken from.
        """
        values = self.parameter_draws.get(parameter)
        if values is None:
            return None
        ok = np.array([s == "ok" for s in self.status[kind]])
        v = values[: ok.size][ok]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        lo, hi = np.quantile(v, quantiles)
        return float(lo), float(hi)

    def counts(self, kind: str) -> dict[str, int]:
        s = self.status[kind]
        return {name: s.count(name) for name in STATUSES}

    def failure_share(self, kind: str | None = None) -> float:
        """Share of draws that were not calibrated or failed the solver gate."""
        kinds = [kind] if kind else list(self.status)
        total = sum(len(self.status[k]) for k in kinds)
        bad = sum(len(self.status[k]) - self.counts(k)["ok"] for k in kinds)
        return bad / total if total else 0.0

    def bands(
        self, kind: str | None = None, quantiles: Sequence[float] = (0.05, 0.5, 0.95)
    ) -> np.ndarray:
        """Quantiles of the merging firms' average price change (``kind=None``: ensemble).

        The ensemble weights each demand form equally over its successful draws.
        """
        if kind is not None:
            v = self.party_change[kind]
            v = v[np.isfinite(v)]
            return np.quantile(v, quantiles) if v.size else np.full(len(quantiles), np.nan)
        values, weights = [], []
        for k in self.party_change:
            v = self.party_change[k]
            v = v[np.isfinite(v)]
            if v.size:
                values.append(v)
                weights.append(np.full(v.size, 1.0 / v.size))
        if not values:
            return np.full(len(quantiles), np.nan)
        return weighted_quantile(np.concatenate(values), np.concatenate(weights), quantiles)

    def cmcr_bands(
        self, kind: str | None = None, quantiles: Sequence[float] = (0.05, 0.5, 0.95)
    ) -> np.ndarray:
        """Quantiles of the merging firms' average compensating marginal cost reduction."""
        src = self.party_cmcr
        if kind is not None:
            v = src[kind][np.isfinite(src[kind])]
            return np.quantile(v, quantiles) if v.size else np.full(len(quantiles), np.nan)
        values, weights = [], []
        for k in src:
            v = src[k][np.isfinite(src[k])]
            if v.size:
                values.append(v)
                weights.append(np.full(v.size, 1.0 / v.size))
        if not values:
            return np.full(len(quantiles), np.nan)
        return weighted_quantile(np.concatenate(values), np.concatenate(weights), quantiles)


def weighted_quantile(
    values: FloatArray, weights: FloatArray, quantiles: Sequence[float]
) -> FloatArray:
    """Quantiles of a weighted sample (inverse of the weighted empirical distribution)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(quantiles, dtype=float), cum, v)


def _default_diversion(market: Market) -> FloatArray:
    """Share-proportional quantity diversions: quantity shares, outside good included."""
    if market.shares.has_outside:
        s = market.as_basis(ShareBasis.QUANTITY).shares.values
    else:
        s = market.quantity_shares_within()
    d = s[None, :] / (1.0 - s[:, None])
    np.fill_diagonal(d, 0.0)
    return d


def _draw_diversion(market: Market, concentration: float, rng: np.random.Generator) -> Diversion:
    base = _default_diversion(market)
    n = base.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        others = np.array([j for j in range(n) if j != i])
        alpha = base[i, others].copy()
        slack = 1.0 - alpha.sum()
        if slack > 1e-9:
            alpha = np.append(alpha, slack)
        draw = rng.dirichlet(concentration * alpha)
        out[i, others] = draw[: others.size]
    return Diversion.quantity(out)


def run_monte_carlo(
    market: Market,
    post: Ownership,
    specs: Sequence[ModelSpec],
    priors: Priors | None = None,
    draws: int = 500,
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
    efficiency: float = 0.0,
    tol: float = DEFAULT_GATE,
) -> MonteCarloResult:
    """Monte Carlo over margins, diversions and demand parameters for each demand form.

    Pass either ``seed`` or an explicit ``rng``. For every draw all models see the same
    perturbed margins, so differences between models in a draw reflect the demand form.
    """
    priors = priors or Priors()
    if draws < 1:
        raise ValueError("draws must be at least 1")
    gen = rng if rng is not None else np.random.default_rng(seed)
    n = market.n_products
    changed = market.ownership.merging_products(post)
    if not changed.any():
        raise ValueError("the post-merger ownership does not differ from the pre-merger one")
    parties = tuple(int(i) for i in np.flatnonzero(changed))
    p0 = market.prices
    weights = p0 * market.quantities()
    base_margin = market.lerner
    known = np.isfinite(base_margin)
    kinds = [s.kind for s in specs]
    if len(set(kinds)) != len(kinds):
        raise ValueError("each demand form may appear once in a Monte Carlo run")
    price_change = {k: np.full((draws, n), np.nan) for k in kinds}
    party_change = {k: np.full(draws, np.nan) for k in kinds}
    party_cmcr = {k: np.full(draws, np.nan) for k in kinds}
    status: dict[str, list[str]] = {k: [] for k in kinds}
    sigma_draws = np.full(draws, np.nan)
    eps_draws = np.full(draws, np.nan)
    w_party = weights[list(parties)]
    uses_diversion = any(spec.kind == "linear" for spec in specs)
    eff = np.zeros(n)
    eff[changed] = efficiency

    for d in range(draws):
        m = base_margin.copy()
        shift = gen.uniform(-priors.margin_halfwidth, priors.margin_halfwidth, n)
        m[known] = np.clip(base_margin[known] + shift[known], 0.02, 0.95)
        sigma = None if priors.sigma_range is None else float(gen.uniform(*priors.sigma_range))
        eps_m = (
            None
            if priors.market_elasticity_range is None
            else float(gen.uniform(*priors.market_elasticity_range))
        )
        sigma_draws[d] = np.nan if sigma is None else sigma
        eps_draws[d] = np.nan if eps_m is None else eps_m
        div = None
        if priors.diversion_concentration is not None and uses_diversion:
            try:
                div = _draw_diversion(market, priors.diversion_concentration, gen)
            except ValueError:
                div = None  # shares cannot be put on a quantity basis: linear fails calibration
        draw_market = replace(market, margins=Margins.lerner(m))
        for spec in specs:
            changes: dict[str, Any] = {}
            if spec.kind == "nested_logit" and sigma is not None:
                changes["sigma"] = sigma
            if spec.kind == "pcaids" and eps_m is not None:
                changes["market_elasticity"] = eps_m
            if spec.kind == "linear" and div is not None:
                changes["diversion"] = div
            this = replace(spec, **changes) if changes else spec
            try:
                cal = calibrate(this, draw_market)
            except (CalibrationError, ValueError):
                status[spec.kind].append("calibration_failed")
                continue
            try:
                eq = solve_bertrand(cal.demand, cal.costs * (1.0 - eff), post, p0, tol=tol)
            except EquilibriumNotFound:
                status[spec.kind].append("solver_failed")
                continue
            status[spec.kind].append("ok")
            change = eq.prices / p0 - 1.0
            price_change[spec.kind][d] = change
            party_change[spec.kind][d] = float(
                np.sum(w_party * change[list(parties)]) / w_party.sum()
            )
            cm = cmcr_from_demand(cal.demand, p0, cal.costs, post)
            party_cmcr[spec.kind][d] = float(
                np.sum(w_party * cm.relative[list(parties)]) / w_party.sum()
            )
    return MonteCarloResult(
        tuple(specs),
        parties,
        weights,
        price_change,
        party_change,
        party_cmcr,
        status,
        draws,
        seed,
        priors,
        {"sigma": sigma_draws, "market_elasticity": eps_draws},
    )
