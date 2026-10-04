"""Derived, display-ready numbers shared by the terminal output and the HTML memo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .case import Analysis
from .metrics import HMTest
from .simulate import ModelOutcome


@dataclass(frozen=True)
class ModelRow:
    """One demand form: point estimates, Monte Carlo band and remedies."""

    kind: str
    label: str
    status: str
    message: str | None
    party_changes: tuple[float, ...] = ()
    average_change: float | None = None
    first_order_average: float | None = None
    band: tuple[float, float, float] | None = None
    cmcr_average: float | None = None
    cmcr_band: tuple[float, float, float] | None = None
    synergy_total: float | None = None
    consumer_harm: float | None = None
    consumer_harm_share: float | None = None
    party_surplus_change: float | None = None
    outsider_changes: tuple[float, ...] = ()
    residual: float | None = None
    parameters: dict[str, float] | None = None
    margin_residual: float | None = None
    notes: tuple[str, ...] = ()
    hm: HMTest | None = None


@dataclass(frozen=True)
class Summary:
    rows: list[ModelRow]
    ensemble_band: tuple[float, float, float] | None
    ensemble_cmcr_band: tuple[float, float, float] | None
    point_range: tuple[float, float] | None
    cmcr_range: tuple[float, float] | None
    synergy_range: tuple[float, float] | None
    harm_range: tuple[float, float] | None
    failure_share: float | None
    mc_draws: int
    absolute_scale: bool


def _weighted(values: np.ndarray, weights: np.ndarray, idx: list[int]) -> float:
    w = weights[idx]
    return float(np.sum(w * values[idx]) / w.sum())


def _row(an: Analysis, o: ModelOutcome) -> ModelRow:
    sim = an.simulation
    idx = list(sim.parties)
    weights = sim.weights
    mc = an.monte_carlo
    kind = o.spec.kind
    band = cmcr_band = None
    if mc is not None and kind in mc.party_change:
        b = mc.bands(kind)
        band = (float(b[0]), float(b[1]), float(b[2])) if np.all(np.isfinite(b)) else None
        cb = mc.cmcr_bands(kind)
        cmcr_band = (float(cb[0]), float(cb[1]), float(cb[2])) if np.all(np.isfinite(cb)) else None
    if not o.ok or o.price_change is None or o.cmcr is None or o.calibration is None:
        return ModelRow(kind, o.spec.label, o.status, o.message, band=band, cmcr_band=cmcr_band)
    outsiders = [i for i in range(len(o.price_change)) if i not in idx]
    harm_share = (
        o.consumer_harm / sim.market.inside_revenue if o.consumer_harm is not None else None
    )
    params = {k: float(v) for k, v in o.calibration.parameters.items()}
    return ModelRow(
        kind,
        o.spec.label,
        "ok",
        None,
        tuple(float(o.price_change[i]) for i in idx),
        _weighted(o.price_change, weights, idx),
        _weighted(o.first_order_change, weights, idx) if o.first_order_change is not None else None,
        band,
        _weighted(o.cmcr.relative, weights, idx),
        cmcr_band,
        o.required_synergy_total,
        o.consumer_harm,
        harm_share,
        o.parties_surplus_change,
        tuple(float(o.price_change[i]) for i in outsiders),
        o.residual,
        params,
        o.calibration.max_margin_residual,
        o.calibration.notes,
        an.hypothetical_monopolist.get(kind),
    )


def summarize(an: Analysis) -> Summary:
    """Collect rows per demand form and ranges across forms."""
    rows = [_row(an, o) for o in an.simulation.outcomes]
    ok = [r for r in rows if r.status == "ok"]
    mc = an.monte_carlo
    ens = ens_c = None
    if mc is not None:
        b = mc.bands()
        if np.all(np.isfinite(b)):
            ens = (float(b[0]), float(b[1]), float(b[2]))
        cb = mc.cmcr_bands()
        if np.all(np.isfinite(cb)):
            ens_c = (float(cb[0]), float(cb[1]), float(cb[2]))

    def rng(vals: list[float]) -> tuple[float, float] | None:
        return (min(vals), max(vals)) if vals else None

    return Summary(
        rows,
        ens,
        ens_c,
        rng([r.average_change for r in ok if r.average_change is not None]),
        rng([r.cmcr_average for r in ok if r.cmcr_average is not None]),
        rng([r.synergy_total for r in ok if r.synergy_total is not None]),
        rng([r.consumer_harm for r in ok if r.consumer_harm is not None]),
        mc.failure_share() if mc is not None else None,
        mc.draws if mc is not None else 0,
        an.case.market.revenue is not None,
    )
