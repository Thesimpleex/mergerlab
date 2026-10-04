"""Why the solver gate exists: the textbook fixed point returns non-equilibria.

The iteration ``p <- c - (Omega * J')^{-1} q`` is the standard way to compute post-merger
prices. For nested logit it often fails to converge, and a loop that merely stops after a
fixed number of iterations returns a price vector that violates the first-order
conditions. This script compares it with the gated solver on (a) the pyblp reference
scenarios and (b) random nested-logit mergers.

Design of (b): six single-product firms, firms 0 and 1 merge; price coefficient
alpha ~ U(1, 3), nesting parameter sigma ~ U(0, 0.9), three nests assigned uniformly at random,
mean utilities delta ~ U(0.5, 2.5), marginal costs ~ U(0.4, 1.0); markets without a pre-merger
equilibrium are redrawn. The baseline is the *undamped* textbook iteration with 10,000
iterations; a damped variant (step 0.5, same iteration count) is reported alongside, so the
failure rate is not read as a statement about every fixed-point scheme.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mergerlab import EquilibriumNotFound, NestedLogit, Ownership, solve_bertrand
from mergerlab.supply import markup_matrix, scaled_residual
from mergerlab.viz import callout, exhibit, save_figure

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "pyblp_merger.json"
GATE = 1e-10
NAIVE_ITERATIONS = 10000


def naive_fixed_point(demand, costs, omega, start, max_iter: int = NAIVE_ITERATIONS):
    """Textbook iteration; returns the last iterate whether or not it converged."""
    p = np.array(start, dtype=float)
    for _ in range(max_iter):
        p_new = costs - np.linalg.solve(markup_matrix(demand, p, omega), demand.quantities(p))
        if not np.all(np.isfinite(p_new)):
            break
        if np.max(np.abs(p_new - p)) < 1e-14:
            return p_new
        p = p_new
    return p


def damped_fixed_point(demand, costs, omega, start, damping=0.5, max_iter=NAIVE_ITERATIONS):
    """Textbook iteration with a convex-combination step ``p + damping (p_new - p)``."""
    p = np.array(start, dtype=float)
    for _ in range(max_iter):
        p_new = costs - np.linalg.solve(markup_matrix(demand, p, omega), demand.quantities(p))
        if not np.all(np.isfinite(p_new)):
            break
        step = damping * (p_new - p)
        if np.max(np.abs(step)) < 1e-14:
            return p + step
        p = p + step
    return p


def _scenario_demand(sc: dict[str, Any]) -> NestedLogit:
    return NestedLogit(
        np.array(sc["delta"]),
        sc["alpha"],
        sc["sigma"] or 0.0,
        sc["nests"] or [0] * len(sc["delta"]),
    )


def iteration_trace(n_iter: int = 40) -> dict[str, Any]:
    """Path of the largest price increase under the naive and the Morrow-Skerlos iteration.

    Scenario: pyblp nested logit with nesting parameter 0.6 (merger of products 0 and 1 in the
    same nest). Both iterations start from the same prices, marginal cost plus one.
    """
    sc = next(s for s in json.loads(FIXTURE.read_text())["scenarios"] if s["sigma"] == 0.6)
    demand = _scenario_demand(sc)
    costs = np.array(sc["costs"])
    post = Ownership.from_owners(sc["owners_post"])
    omega = post.matrix
    pre = np.array(sc["pre"]["prices"])
    naive = zeta = costs + 1.0
    out: dict[str, list[float]] = {
        "naive_max_increase": [],
        "naive_residual": [],
        "zeta_max_increase": [],
        "zeta_residual": [],
    }
    for _ in range(n_iter + 1):
        out["naive_max_increase"].append(float(np.max(naive / pre - 1)))
        out["naive_residual"].append(scaled_residual(demand, naive, costs, post))
        out["zeta_max_increase"].append(float(np.max(zeta / pre - 1)))
        out["zeta_residual"].append(scaled_residual(demand, zeta, costs, post))
        naive = costs - np.linalg.solve(
            markup_matrix(demand, naive, omega), demand.quantities(naive)
        )
        lam, gamma = demand.zeta_terms(zeta)
        zeta = costs + (demand.quantities(zeta) + (omega * gamma.T) @ (zeta - costs)) / lam
    target = float(np.max(np.array(sc["post"]["prices"]) / pre - 1))
    return {"scenario": sc["name"], "pyblp_max_increase": target, **out}


def run(n_random: int = 400, seed: int = 5) -> dict[str, Any]:
    fixtures = json.loads(FIXTURE.read_text())
    scenarios = []
    for sc in fixtures["scenarios"]:
        demand = _scenario_demand(sc)
        costs = np.array(sc["costs"])
        post = Ownership.from_owners(sc["owners_post"])
        pre_prices = np.array(sc["pre"]["prices"])
        pyblp_effect = float(np.max(np.array(sc["post"]["prices"]) / pre_prices - 1))
        naive = naive_fixed_point(demand, costs, post.matrix, costs + 1.0)
        eq = solve_bertrand(demand, costs, post, pre_prices)
        scenarios.append(
            {
                "name": sc["name"],
                "sigma": sc["sigma"],
                "pyblp_effect": pyblp_effect,
                "gated_effect": float(np.max(eq.prices / pre_prices - 1)),
                "gated_residual": eq.residual,
                "naive_effect": float(np.max(naive / pre_prices - 1)),
                "naive_residual": scaled_residual(demand, naive, costs, post),
                "naive_passes_gate": bool(scaled_residual(demand, naive, costs, post) < GATE),
            }
        )
    rng = np.random.default_rng(seed)
    naive_fail = gated_fail = total = naive_wrong = damped_fail = 0
    errors = []
    while total < n_random:
        j = 6
        alpha = rng.uniform(1.0, 3.0)
        sigma = rng.uniform(0.0, 0.9)
        nests = rng.integers(0, 3, j)
        delta = rng.uniform(0.5, 2.5, j)
        costs = rng.uniform(0.4, 1.0, j)
        demand = NestedLogit(delta, alpha, sigma, nests.tolist())
        owners = [f"F{i}" for i in range(j)]
        pre, post = (
            Ownership.from_owners(owners),
            Ownership.from_owners(owners).merged(["F0", "F1"]),
        )
        try:
            base = solve_bertrand(demand, costs, pre, costs + 1.0)
        except EquilibriumNotFound:
            continue
        total += 1
        try:
            eq = solve_bertrand(demand, costs, post, base.prices)
        except EquilibriumNotFound:
            gated_fail += 1
            continue
        damped = damped_fixed_point(demand, costs, post.matrix, base.prices)
        if scaled_residual(demand, damped, costs, post) >= GATE:
            damped_fail += 1
        naive = naive_fixed_point(demand, costs, post.matrix, base.prices)
        if scaled_residual(demand, naive, costs, post) >= GATE:
            naive_fail += 1
            errors.append(float(np.max(np.abs(naive / eq.prices - 1))))
        elif np.max(np.abs(naive / eq.prices - 1)) > 1e-6:
            naive_wrong += 1
    return {
        "scenarios": scenarios,
        "trace": iteration_trace(),
        "random": {
            "markets": n_random,
            "seed": seed,
            "naive_fails_gate": naive_fail,
            "naive_fail_share": naive_fail / n_random,
            "gated_fails": gated_fail,
            "gated_fail_share": gated_fail / n_random,
            "naive_converged_to_other_root": naive_wrong,
            "median_price_error_when_naive_fails": float(np.median(errors)) if errors else 0.0,
            "naive_iterations": NAIVE_ITERATIONS,
            "damped_fails_gate": damped_fail,
            "damped_fail_share": damped_fail / n_random,
            "damping": 0.5,
            "design": (
                "6 single-product firms, firms 0 and 1 merge; alpha ~ U(1, 3), sigma ~ U(0, 0.9), "
                "3 nests drawn uniformly, delta ~ U(0.5, 2.5), costs ~ U(0.4, 1.0)"
            ),
        },
    }


EXHIBIT_ITERATION = 4
EXHIBIT = 5
LABELS = {
    "logit_6_merge_01": "Logit, 6 products,\nmerger of two",
    "logit_5_multiproduct": "Logit, 5 products,\nmulti-product firm",
    "nested_0.6_6_merge_01_same_nest": "Nested logit (0.6),\nmerger within a nest",
    "nested_0.3_6_merge_across_nests": "Nested logit (0.3),\nmerger across nests",
    "nested_0.85_7_three_nests": "Nested logit (0.85),\nthree nests",
}


def plot(res: dict[str, Any]) -> list[Path]:
    """Exhibit 5: equilibrium and textbook-iteration price increase in the pyblp scenarios."""
    sc = sorted(res["scenarios"], key=lambda s: -s["gated_effect"])
    n = len(sc)
    agreement = max(abs(s["gated_effect"] - s["pyblp_effect"]) for s in sc)
    assert agreement < 1e-13
    pyblp_version = json.loads(FIXTURE.read_text())["pyblp_version"]
    failing = [s for s in sc if not s["naive_passes_gate"]]
    worst = max(failing, key=lambda s: s["naive_effect"] - s["gated_effect"])
    y_worst = n - sc.index(worst)

    def build(p: dict[str, str]):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        halo = {"boxstyle": "square,pad=0.12", "fc": p["surface"], "ec": "none"}
        for i, s in enumerate(sc):
            y = n - i
            g, t = 100 * s["gated_effect"], 100 * s["naive_effect"]
            dot = {"ms": 7.5, "mec": p["surface"], "mew": 1.4}
            if s["naive_passes_gate"]:
                ax.plot(g, y, "o", color=p["ink_secondary"], zorder=5, **dot)
                ax.annotate(
                    f"{g:.1f}%",
                    (g, y),
                    xytext=(10, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=p["ink"],
                    bbox=halo,
                )
                ax.annotate(
                    "iteration agrees",
                    (g, y),
                    xytext=(50, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8.5,
                    color=p["muted"],
                    bbox=halo,
                )
                continue
            ax.plot([g, t], [y, y], color=p["rule"], lw=2.4, solid_capstyle="round", zorder=2)
            ax.plot(g, y, "o", color=p["ink_secondary"], zorder=5, **dot)
            ax.plot(t, y, "o", color=p["accent"], zorder=5, **dot)
            for value, colour, other in ((g, p["ink"], t), (t, p["accent"], g)):
                if s is worst and value == t:
                    continue
                lower = value < other
                crowded = lower and value < 20  # a label to the left would touch the axis
                ax.annotate(
                    f"{value:.1f}%",
                    (value, y),
                    xytext=(0, 10) if crowded else (-10 if lower else 10, 0),
                    textcoords="offset points",
                    ha="center" if crowded else ("right" if lower else "left"),
                    va="bottom" if crowded else "center",
                    fontsize=9,
                    color=colour,
                    fontweight="medium" if colour == p["accent"] else "regular",
                    bbox=halo,
                )
        ax.set_yticks(range(n, 0, -1))
        ax.set_yticklabels(
            [LABELS.get(s["name"], s["name"]) for s in sc], fontsize=9, color=p["ink_secondary"]
        )
        ax.set_ylim(0.45, n + 0.55)
        ax.set_xlim(0, 165)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xticklabels(["0", "50", "100", "150%"])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("Largest post-merger price increase")
        callout(
            ax,
            p,
            (100 * worst["naive_effect"], y_worst),
            f"{100 * worst['naive_effect']:.1f}%",
            f"Textbook iteration;\nthe equilibrium is {100 * worst['gated_effect']:.1f}%",
            dx=26,
            dy=-30,
            ha="right",
        )
        ax.text(
            0.02,
            0.985,
            "Grey dot: gated solver, equals pyblp",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color=p["ink_secondary"],
            bbox=halo,
        )
        ax.text(
            0.02,
            0.885,
            "Red dot: textbook iteration, fails the gate",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color=p["accent"],
            bbox=halo,
        )
        exhibit(
            fig,
            p,
            kicker=f"Exhibit {EXHIBIT} · Solver gate",
            title=(
                f"In {len(failing)} of {n} pyblp scenarios the textbook iteration ends away from "
                "the equilibrium, above or below it"
            ),
            subtitle="Largest post-merger price increase of any product, by method, %",
            source=(
                f"Source: pyblp {pyblp_version} reference equilibria; textbook iteration: "
                f"{res['random']['naive_iterations']:,} undamped iterations from marginal cost "
                "plus one."
            ),
            wrap=60,
            right_in=0.6,
        )
        fig.subplots_adjust(left=2.0 / 8.0)
        return fig

    return save_figure(build, ROOT / "docs" / "figures" / "solver_gate")


def print_table(res: dict[str, Any]) -> None:
    console = Console(highlight=False)
    console.print(Text("Post-merger equilibrium: textbook iteration vs gated solver", style="bold"))
    console.print(
        Text(
            f"pyblp reference scenarios; gate {GATE:g}; textbook = {NAIVE_ITERATIONS:,} iterations",
            style="dim",
        )
    )
    console.print()
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Scenario", no_wrap=True)
    table.add_column("pyblp", justify="right")
    table.add_column("mergerlab", justify="right")
    table.add_column("Textbook", justify="right")
    table.add_column("Textbook FOC residual", justify="right")
    table.add_column("Textbook vs gate")
    for s in res["scenarios"]:
        ok = s["naive_passes_gate"]
        table.add_row(
            LABELS.get(s["name"], s["name"]).replace("\n", " "),
            f"{100 * s['pyblp_effect']:+.1f}%",
            f"{100 * s['gated_effect']:+.1f}%",
            f"{100 * s['naive_effect']:+.1f}%",
            f"{s['naive_residual']:.1e}",
            Text("passes" if ok else "FAILS", style="green" if ok else "red"),
        )
    console.print(table)
    r = res["random"]
    console.print()
    console.print(
        Text.assemble(
            ("Random nested-logit mergers  ", "bold"),
            (f"{r['markets']} markets, seed {r['seed']}", "dim"),
        )
    )
    console.print(
        Text.assemble(
            (f"textbook iteration fails the gate in {r['naive_fails_gate']}", "red"),
            ("; ", "dim"),
            (f"gated solver finds an equilibrium in {r['markets'] - r['gated_fails']}", "green"),
            (f" of {r['markets']}", "dim"),
        )
    )
    console.print(
        Text(
            f"a damped iteration (step {r['damping']:g}) fails in {r['damped_fails_gate']}; "
            f"design: alpha U(1, 3), sigma U(0, 0.9), 3 nests",
            style="dim",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="store_true", help="print the rich table only")
    parser.add_argument(
        "--results", type=Path, help="with --table: read this results.json instead of recomputing"
    )
    args = parser.parse_args()
    if args.table and args.results is not None:
        print_table(json.loads(args.results.read_text())["solver_gate"])
        raise SystemExit(0)
    out = run()
    if args.table:
        print_table(out)
    else:
        plot(out)
        print(json.dumps(out["random"], indent=1))
