"""Helpers shared by the test modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mergerlab.units import FloatArray

DATA = Path(__file__).parent / "data"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((DATA / name).read_text())


def nan_list(values: list[float | None]) -> list[float]:
    return [np.nan if v is None else float(v) for v in values]


def numerical_jacobian(fun, p: FloatArray, h: float = 1e-6) -> FloatArray:
    """Central-difference Jacobian ``d fun_j / d p_k``."""
    n = p.size
    out = np.empty((fun(p).size, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h * max(1.0, abs(p[k]))
        out[:, k] = (fun(p + e) - fun(p - e)) / (2 * e[k])
    return out


def naive_fixed_point(demand, costs, omega, p0, max_iter=10000):
    """The textbook iteration ``p <- c - (Omega * J')^{-1} q`` (no convergence guarantee)."""
    from mergerlab.supply import markup_matrix

    p = np.array(p0, dtype=float)
    for _ in range(max_iter):
        p_new = costs - np.linalg.solve(markup_matrix(demand, p, omega), demand.quantities(p))
        if np.max(np.abs(p_new - p)) < 1e-14:
            return p_new
        p = p_new
    return p


def random_logit_market(rng, n_products: int = 5, firms: int | None = None):
    """Random logit-consistent market with known outside share and two margins."""
    from mergerlab.market import Market
    from mergerlab.units import Margins, Shares

    shares = rng.dirichlet(np.ones(n_products + 1))
    inside = shares[:-1]
    prices = rng.uniform(0.8, 2.5, n_products)
    margins = np.full(n_products, np.nan)
    margins[0] = rng.uniform(0.2, 0.7)
    n_firms = firms if firms is not None else n_products
    owners = [f"F{i % n_firms}" for i in range(n_products)]
    return Market.build(
        prices,
        Shares.total(inside, "quantity"),
        Margins.lerner(margins),
        owners,
    )
