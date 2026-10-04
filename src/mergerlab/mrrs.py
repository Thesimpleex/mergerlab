"""Monte Carlo design of Miller, Remer, Ryan and Sheu (2017), logit and linear demand.

Miller, Remer, Ryan and Sheu, "Upward pricing pressure as a predictor of merger price
effects", International Journal of Industrial Organization 52 (2017), 216-247. Each
market has six single-product firms and an outside good, unit prices, a margin for firm 1
drawn from U(0.2, 0.8), and shares that rationalise a logit demand system; firms 1 and 2
merge. The paper does not state how the shares are drawn. Independent U(0, 1) draws for the
six firms and the outside good, normalised to sum to one, reproduce its Table 1 order
statistics for shares and the HHI (see ``scripts/exp_mrrs.py``), so that design is used.
Markets whose margins cannot be rationalised by logit demand are redrawn, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demand import CalibrationError, Linear
from .demand.logit import Logit
from .errors import EquilibriumNotFound
from .metrics import first_order_price_effects, pricing_pressure
from .supply import recover_costs, solve_bertrand
from .units import FloatArray, Ownership

N_FIRMS = 6


@dataclass(frozen=True)
class MillerDraws:
    """Per-market statistics of the merger of firms 1 and 2."""

    share1: FloatArray
    margin1: FloatArray
    diversion12: FloatArray
    hhi_pre: FloatArray
    hhi_post: FloatArray
    delta_hhi: FloatArray
    upp: FloatArray
    effect_logit: FloatArray
    effect_linear: FloatArray
    foa_logit: FloatArray
    foa_linear: FloatArray
    redraws: int
    solver_failures: int

    def __len__(self) -> int:
        return int(self.share1.size)


def draw_market(rng: np.random.Generator) -> tuple[FloatArray, float]:
    """Shares of six firms and the outside good, and firm 1's margin."""
    u = rng.uniform(size=N_FIRMS + 1)
    return u / u.sum(), float(rng.uniform(0.2, 0.8))


def simulate_markets(n_markets: int = 4500, seed: int = 2017) -> MillerDraws:
    """Simulate ``n_markets`` rationalisable markets (see the module docstring)."""
    rng = np.random.default_rng(seed)
    prices = np.ones(N_FIRMS)
    pre = Ownership.from_owners([f"F{i}" for i in range(N_FIRMS)])
    post = pre.merged(["F0", "F1"])
    cols: dict[str, list[float]] = {
        k: []
        for k in (
            "share1",
            "margin1",
            "diversion12",
            "hhi_pre",
            "hhi_post",
            "delta_hhi",
            "upp",
            "effect_logit",
            "effect_linear",
            "foa_logit",
            "foa_linear",
        )
    }
    redraws = failures = 0
    while len(cols["share1"]) < n_markets:
        shares, m1 = draw_market(rng)
        s, s0 = shares[:N_FIRMS], shares[N_FIRMS]
        alpha = 1.0 / (m1 * prices[0] * (1.0 - s[0]))
        logit = Logit(np.log(s / s0) + alpha * prices, alpha)
        try:
            costs = recover_costs(logit, prices, pre)
        except CalibrationError:
            redraws += 1
            continue
        try:
            eq = solve_bertrand(logit, costs, post, prices)
            jac = logit.jacobian(prices)
            linear = Linear(s - jac @ prices, jac, prices)
            eq_lin = solve_bertrand(linear, costs, post, prices)
        except EquilibriumNotFound:
            failures += 1
            continue
        div12 = float(-jac[1, 0] / jac[0, 0])
        pressure = pricing_pressure(logit, prices, costs, pre, post)
        foa = first_order_price_effects(logit, prices, costs, pre, post)
        foa_lin = first_order_price_effects(linear, prices, costs, pre, post)
        cols["share1"].append(float(s[0]))
        cols["margin1"].append(m1)
        cols["diversion12"].append(div12)
        cols["hhi_pre"].append(1e4 * float(np.sum(s**2)))
        cols["hhi_post"].append(1e4 * float(np.sum(s**2) + 2 * s[0] * s[1]))
        cols["delta_hhi"].append(2e4 * float(s[0] * s[1]))
        cols["upp"].append(float(pressure[0]))
        cols["effect_logit"].append(float(eq.prices[0] - 1.0))
        cols["effect_linear"].append(float(eq_lin.prices[0] - 1.0))
        cols["foa_logit"].append(float(foa[0]))
        cols["foa_linear"].append(float(foa_lin[0]))
    return MillerDraws(
        **{k: np.asarray(v) for k, v in cols.items()}, redraws=redraws, solver_failures=failures
    )
