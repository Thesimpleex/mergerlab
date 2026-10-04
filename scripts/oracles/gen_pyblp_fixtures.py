"""Generate pyblp reference equilibria for logit and nested-logit merger simulations.

pyblp (Conlon and Gortmaker, 2020) is used only here, to produce
``tests/data/pyblp_merger.json``; the test suite reads the JSON and never imports pyblp.

Run from the repository root with the ``oracle`` extra installed:

    python scripts/oracles/gen_pyblp_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyblp

pyblp.options.verbose = False

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "pyblp_merger.json"

ITERATION = pyblp.Iteration("simple", {"atol": 1e-14, "max_evaluations": 100000})


def equilibrium(
    x: np.ndarray,
    costs: np.ndarray,
    firms: np.ndarray,
    nests: np.ndarray | None,
    alpha: float,
    beta_x: float,
    rho: float | None,
) -> dict[str, list[float]]:
    n = x.size
    data: dict[str, np.ndarray] = {
        "market_ids": np.zeros(n),
        "firm_ids": firms,
        "x": x,
        "prices": np.ones(n),
        "shares": np.full(n, 0.05),
    }
    if nests is not None:
        data["nesting_ids"] = nests
    sim = pyblp.Simulation(
        pyblp.Formulation("0 + prices + x"),
        data,
        beta=[-alpha, beta_x],
        xi=np.zeros(n),
        rho=rho,
        seed=0,
    )
    res = sim.replace_endogenous(costs=costs, iteration=ITERATION)
    return {
        "prices": np.asarray(res.product_data.prices).ravel().tolist(),
        "shares": np.asarray(res.product_data.shares).ravel().tolist(),
    }


def scenario(
    name: str,
    seed: int,
    n: int,
    alpha: float,
    rho: float | None,
    nests: list[int] | None,
    pre_firms: list[int],
    post_firms: list[int],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    beta_x = 2.0
    x = rng.uniform(1.0, 2.0, n)
    costs = rng.uniform(0.5, 1.0, n)
    nest_arr = None if nests is None else np.asarray(nests)
    pre = equilibrium(x, costs, np.asarray(pre_firms), nest_arr, alpha, beta_x, rho)
    post = equilibrium(x, costs, np.asarray(post_firms), nest_arr, alpha, beta_x, rho)
    return {
        "name": name,
        "delta": (beta_x * x).tolist(),
        "alpha": alpha,
        "sigma": rho,
        "nests": nests,
        "costs": costs.tolist(),
        "owners_pre": pre_firms,
        "owners_post": post_firms,
        "pre": pre,
        "post": post,
    }


def main() -> None:
    scenarios = [
        scenario("logit_6_merge_01", 1, 6, 1.5, None, None, [0, 1, 2, 3, 4, 5], [0, 0, 2, 3, 4, 5]),
        scenario("logit_5_multiproduct", 7, 5, 2.2, None, None, [0, 0, 1, 2, 3], [0, 0, 0, 2, 3]),
        scenario(
            "nested_0.6_6_merge_01_same_nest",
            1,
            6,
            1.5,
            0.6,
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 2, 3, 4, 5],
        ),
        scenario(
            "nested_0.3_6_merge_across_nests",
            3,
            6,
            1.8,
            0.3,
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 0, 4, 5],
        ),
        scenario(
            "nested_0.85_7_three_nests",
            5,
            7,
            1.2,
            0.85,
            [0, 0, 1, 1, 1, 2, 2],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 0, 2, 3, 4, 5, 6],
        ),
    ]
    payload = {
        "generator": "scripts/oracles/gen_pyblp_fixtures.py",
        "pyblp_version": pyblp.__version__,
        "numpy_version": np.__version__,
        "convention": "u_ij = delta_j - alpha p_j + nest term; costs are marginal costs; "
        "market size 1; outside good utility 0",
        "scenarios": scenarios,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT} ({len(scenarios)} scenarios, pyblp {pyblp.__version__})")


if __name__ == "__main__":
    main()
