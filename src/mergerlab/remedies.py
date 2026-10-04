"""Single-product divestiture screening.

For every product of the merging parties, transfer it to an independent owner, recompute the
post-merger equilibrium under the same demand system and calibration, and rank the options by
the consumer harm that remains. This evaluates one product at a time; it is not a remedy
optimiser (the buyer's costs and pricing are those of an independent single-product firm that
inherits the product's marginal cost).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import Market
from .simulate import ModelOutcome, ModelSpec, simulate_one
from .supply import DEFAULT_GATE
from .units import FloatArray, Ownership

__all__ = ["DivestitureOption", "DivestitureReport", "evaluate_divestitures"]


@dataclass(frozen=True)
class DivestitureOption:
    """Outcome when product ``product`` is divested to an independent owner."""

    product: int
    label: str
    outcome: ModelOutcome
    consumer_harm: float | None
    max_price_change: float | None

    @property
    def ok(self) -> bool:
        return self.outcome.ok


@dataclass(frozen=True)
class DivestitureReport:
    """Baseline merger and the divestiture options, best (least remaining harm) first."""

    baseline: ModelOutcome
    options: list[DivestitureOption]

    @property
    def best(self) -> DivestitureOption | None:
        usable = [o for o in self.options if o.ok]
        return usable[0] if usable else None


def evaluate_divestitures(
    spec: ModelSpec,
    market: Market,
    post: Ownership,
    efficiency: float | FloatArray = 0.0,
    candidates: list[int] | None = None,
    tol: float = DEFAULT_GATE,
) -> DivestitureReport:
    """Rank single-product divestitures by the consumer harm that remains.

    Parameters
    ----------
    candidates
        Product indices that may be divested; by default every product whose ownership
        changes in the merger.
    """
    if post.owners is None:
        raise ValueError("divestitures need an ownership built with Ownership.from_owners")
    changed = market.ownership.merging_products(post)
    if not changed.any():
        raise ValueError("the post-merger ownership does not differ from the pre-merger one")
    pool = [int(i) for i in np.flatnonzero(changed)] if candidates is None else list(candidates)
    if any(i not in set(np.flatnonzero(changed).tolist()) for i in pool):
        raise ValueError("candidates must be products of the merging parties")
    baseline = simulate_one(spec, market, post, efficiency, tol)
    p0 = market.prices
    options = []
    for i in pool:
        owners = list(post.owners)
        owners[i] = f"divested:{market.labels[i]}"
        outcome = simulate_one(spec, market, Ownership.from_owners(owners), efficiency, tol)
        top = None if outcome.prices_post is None else float(np.max(outcome.prices_post / p0 - 1))
        options.append(
            DivestitureOption(
                i,
                market.labels[i],
                outcome,
                outcome.consumer_harm,
                top,
            )
        )
    options.sort(key=lambda o: (not o.ok, o.consumer_harm if o.consumer_harm is not None else 0.0))
    return DivestitureReport(baseline, options)
