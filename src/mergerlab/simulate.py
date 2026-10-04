"""Merger simulation across demand systems: calibration, equilibrium and metrics in one call."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .demand import (
    Calibration,
    calibrate_ces,
    calibrate_linear,
    calibrate_logit,
    calibrate_logit_alm,
    calibrate_nested_logit,
    calibrate_pcaids,
)
from .errors import CalibrationError, EquilibriumNotFound
from .market import Market
from .metrics import (
    CMCR,
    cmcr_from_demand,
    compensating_variation,
    first_order_price_effects,
    guppi,
    producer_surplus,
)
from .screens import Concentration, GuidelineResult, concentration, screen
from .supply import DEFAULT_GATE, solve_bertrand
from .units import Diversion, FloatArray, Margins, Ownership, ShareBasis

MODEL_KINDS = ("logit", "logit_alm", "nested_logit", "ces", "linear", "pcaids")
MODEL_LABELS = {
    "logit": "Logit",
    "logit_alm": "Logit (outside share from margins)",
    "nested_logit": "Nested logit",
    "ces": "CES",
    "linear": "Linear",
    "pcaids": "PCAIDS",
}


@dataclass(frozen=True)
class ModelSpec:
    """Which demand system to calibrate, and its parameters beyond shares, prices, margins.

    ``nested_logit`` needs ``nests`` (one label per product) and ``sigma``. ``pcaids`` takes
    ``own_elasticity`` of the anchor product ``known_index`` (default: ``-1 / margin`` of that
    product, valid when it is a single-product firm) and ``market_elasticity``; the anchor is
    reported in the calibration notes. ``linear`` accepts a
    quantity ``diversion`` (default: proportional to shares) and, as an explicit opt-in,
    ``fill_margins_from="logit"`` to take margins that are not supplied from a logit
    calibration (the linear system needs a margin for every product).
    """

    kind: str
    nests: tuple[int, ...] | None = None
    sigma: float | None = None
    own_elasticity: float | None = None
    market_elasticity: float = -1.0
    known_index: int = 0
    diversion: Diversion | None = None
    fill_margins_from: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS:
            raise ValueError(f"unknown demand model {self.kind!r}; choose from {MODEL_KINDS}")
        if self.kind == "nested_logit" and (self.nests is None or self.sigma is None):
            raise ValueError("nested_logit needs nests and sigma")

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.kind]


def calibrate(spec: ModelSpec, market: Market) -> Calibration:
    """Calibrate ``spec`` to ``market``, converting share conventions where that is exact.

    The conversions are the ones each system's definition requires: logit systems use
    quantity shares, CES and PCAIDS revenue shares; shares with an outside good are
    converted with ``market.outside_price``.
    """
    kind = spec.kind
    if kind in ("logit", "nested_logit", "ces") and not market.shares.has_outside:
        raise ValueError(
            f"{MODEL_LABELS[kind]} needs an outside good with a known share; declare "
            "outside_share (or use logit_alm to calibrate it from margins)"
        )
    if kind == "logit":
        return calibrate_logit(market.as_basis(ShareBasis.QUANTITY))
    if kind == "logit_alm":
        return calibrate_logit_alm(market.within().as_basis(ShareBasis.QUANTITY))
    if kind == "nested_logit":
        if spec.nests is None or spec.sigma is None:
            raise ValueError("nested_logit needs nests and sigma")
        return calibrate_nested_logit(market.as_basis(ShareBasis.QUANTITY), spec.nests, spec.sigma)
    if kind == "ces":
        return calibrate_ces(market.as_basis(ShareBasis.REVENUE))
    if kind == "linear":
        m = market.as_basis(ShareBasis.QUANTITY) if market.shares.has_outside else market
        notes: tuple[str, ...] = ()
        if spec.fill_margins_from is not None and not np.all(np.isfinite(m.lerner)):
            if spec.fill_margins_from != "logit":
                raise ValueError("fill_margins_from must be 'logit' or None")
            base = calibrate(ModelSpec("logit"), market)
            missing = ~np.isfinite(m.lerner)
            filled = np.where(missing, base.fitted_margins, m.lerner)
            m = replace(m, margins=Margins.lerner(filled))
            names = ", ".join(market.labels[i] for i in np.flatnonzero(missing))
            notes = (f"margins of {names} taken from the logit calibration",)
        return replace(calibrate_linear(m, spec.diversion), market=market, notes=notes)
    own = spec.own_elasticity
    basis = "given"
    if own is None:
        margin = market.lerner[spec.known_index]
        if not np.isfinite(margin):
            raise CalibrationError(
                "PCAIDS needs own_elasticity in [models.pcaids], or a margin for its anchor "
                "product to derive it"
            )
        if market.ownership.matrix[spec.known_index].sum() != 1:
            raise CalibrationError(
                "-1 / margin is the own elasticity only for a single-product firm; "
                "set own_elasticity in [models.pcaids]"
            )
        own = -1.0 / float(margin)
        basis = f"-1 divided by its Lerner margin of {float(margin):.1%}"
    cal = calibrate_pcaids(
        market.within().as_basis(ShareBasis.REVENUE), own, spec.market_elasticity, spec.known_index
    )
    anchor = (
        f"anchored on {market.labels[spec.known_index]}, own-price elasticity {own:.3g} "
        f"({basis}), market elasticity {spec.market_elasticity:g}"
    )
    notes = (*cal.notes, anchor, "the other slopes follow from proportional diversion")
    return replace(cal, market=market, notes=notes)


@dataclass(frozen=True)
class ModelOutcome:
    """Result of one demand system for one merger (or why it failed)."""

    spec: ModelSpec
    status: str
    message: str | None = None
    calibration: Calibration | None = None
    prices_post: FloatArray | None = None
    price_change: FloatArray | None = None
    quantity_change: FloatArray | None = None
    cmcr: CMCR | None = None
    first_order_change: FloatArray | None = None
    guppi: FloatArray | None = None
    consumer_harm: float | None = None
    producer_surplus_change: float | None = None
    parties_surplus_change: float | None = None
    required_synergy_total: float | None = None
    residual: float | None = None
    method: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class SimulationResult:
    """All results of :func:`simulate_merger`."""

    market: Market
    post: Ownership
    parties: tuple[int, ...]
    concentration: Concentration
    screens: list[GuidelineResult]
    outcomes: list[ModelOutcome]
    efficiency: float
    weights: FloatArray = field(repr=False, default_factory=lambda: np.empty(0))

    def party_price_change(self, outcome: ModelOutcome) -> float | None:
        """Pre-merger-revenue-weighted average price change of the merging firms' products."""
        if outcome.price_change is None:
            return None
        idx = list(self.parties)
        w = self.weights[idx]
        return float(np.sum(w * outcome.price_change[idx]) / w.sum())

    def price_effect_range(self) -> tuple[float, float] | None:
        changes = [self.party_price_change(o) for o in self.outcomes if o.ok]
        vals = [v for v in changes if v is not None]
        return (min(vals), max(vals)) if vals else None


def simulate_one(
    spec: ModelSpec,
    market: Market,
    post: Ownership,
    efficiency: float | FloatArray = 0.0,
    tol: float = DEFAULT_GATE,
) -> ModelOutcome:
    """Calibrate one demand system, solve the post-merger equilibrium and compute metrics.

    Calibration, solver-gate and metric failures are returned as outcomes with ``status``
    ``"calibration_failed"``, ``"solver_failed"`` or ``"metrics_failed"`` and a message;
    they are never silently dropped. A ``"metrics_failed"`` outcome still carries the
    solved prices.
    """
    try:
        cal = calibrate(spec, market)
    except (CalibrationError, ValueError) as exc:
        return ModelOutcome(spec, "calibration_failed", str(exc))
    demand, p0, c0 = cal.demand, market.prices, cal.costs
    changed = market.ownership.merging_products(post)
    eff = np.zeros(p0.size)
    eff[changed] = np.broadcast_to(np.asarray(efficiency, dtype=float), p0.shape)[changed]
    c1 = c0 * (1.0 - eff)
    try:
        eq = solve_bertrand(demand, c1, post, p0, tol=tol)
    except EquilibriumNotFound as exc:
        return ModelOutcome(spec, "solver_failed", str(exc), calibration=cal)
    try:
        q0 = demand.quantities(p0)
        cm = cmcr_from_demand(demand, p0, c0, post)
        div = Diversion.from_demand(demand.diversion_quantity(p0))
        g = guppi(p0, Margins.lerner(cal.fitted_margins), div, market.ownership, post)
        foa = first_order_price_effects(
            demand, p0, c0, market.ownership, post, cost_change=-eff * c0
        )
        ps0 = producer_surplus(demand, p0, c0)
        ps1 = producer_surplus(demand, eq.prices, c1)
        harm = compensating_variation(demand, p0, eq.prices)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return ModelOutcome(
            spec,
            "metrics_failed",
            f"the merger was solved but its metrics could not be computed: {exc}",
            calibration=cal,
            prices_post=eq.prices,
            price_change=eq.prices / p0 - 1.0,
            quantity_change=eq.quantities / demand.quantities(p0) - 1.0,
            residual=eq.residual,
            method=eq.method,
        )
    return ModelOutcome(
        spec,
        "ok",
        None,
        cal,
        eq.prices,
        eq.prices / p0 - 1.0,
        eq.quantities / q0 - 1.0,
        cm,
        foa / p0,
        g,
        harm,
        float(ps1.sum() - ps0.sum()),
        float(ps1[changed].sum() - ps0[changed].sum()),
        float(np.sum(cm.level[changed] * q0[changed])),
        eq.residual,
        eq.method,
    )


def simulate_merger(
    market: Market,
    post: Ownership,
    models: Sequence[ModelSpec],
    efficiency: float | FloatArray = 0.0,
    rulesets: tuple[str, ...] = ("us2023", "us2010", "eu2004"),
    outside: str = "atomistic",
    tol: float = DEFAULT_GATE,
) -> SimulationResult:
    """Screen a merger and simulate it under each demand system in ``models``.

    ``efficiency`` is the proportional marginal-cost saving on the merging firms' products
    (a scalar or one value per product). Each demand system is calibrated to the same
    prices, shares and margins; differences in the predicted price effect are due to the
    demand form alone.
    """
    if market.ownership.n_products != post.n_products:
        raise ValueError("post-merger ownership must cover the same products")
    changed = market.ownership.merging_products(post)
    if not changed.any():
        raise ValueError("the post-merger ownership does not differ from the pre-merger one")
    conc = concentration(market.shares, market.ownership, post, outside=outside)  # type: ignore[arg-type]
    outcomes = [simulate_one(m, market, post, efficiency, tol) for m in models]
    weights = market.prices * market.quantities()
    return SimulationResult(
        market,
        post,
        tuple(int(i) for i in np.flatnonzero(changed)),
        conc,
        screen(conc, rulesets),
        outcomes,
        float(np.max(np.broadcast_to(np.asarray(efficiency, dtype=float), changed.shape)[changed])),
        weights,
    )
