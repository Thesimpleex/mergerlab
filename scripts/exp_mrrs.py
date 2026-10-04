"""Reduced reproduction of Miller, Remer, Ryan and Sheu (2017) on first-order pricing pressure.

Simulates 4,500 six-firm markets with logit-rationalisable shares and a margin for firm 1, merges
firms 1 and 2, and compares the upward pricing pressure ``D_12 m_2`` (GUPPI scale, prices
normalised to one) and the Jaffe-Weyl first-order approximation with the full merger simulation
under logit and linear demand. The order statistics are compared with the paper's Table 1 and the
median absolute prediction errors with its Table 2.

The first-order approximation is evaluated with the pass-through of the demand system that
generated the truth, which pricing pressure (a shortcut that needs no curvature information)
does not have, so its small error is not a like-for-like comparison with pricing pressure. The
comparable benchmarks for pricing pressure are the first-order approximation computed from a
*misspecified* pass-through (linear when logit is the truth, and the reverse) and, equivalently
for the logit-truth case, a simulation under the wrong demand form: under linear demand the
first-order approximation is exact, so the two coincide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mergerlab.mrrs import simulate_markets
from mergerlab.viz import callout, exhibit, save_figure

ROOT = Path(__file__).resolve().parents[1]
N_MARKETS = 4500
SEED = 2017

# Table 1 (medians) and Table 2 (UPP and logit-simulation rows) of the paper.
PUBLISHED = {
    "share1": 0.15,
    "margin1": 0.49,
    "diversion12": 0.17,
    "hhi_pre": 1562.0,
    "hhi_post": 1931.0,
    "delta_hhi": 317.0,
    "upp": 0.07,
    "effect_logit": 0.06,
    "effect_linear": 0.05,
}
PUBLISHED_MAPE = {
    "upp_vs_logit": 0.006,
    "upp_vs_linear": 0.022,
    "logit_simulation_vs_linear": 0.014,
    "linear_simulation_vs_logit": 0.014,
}


def _mape(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(np.abs(a - b)))


def run(n_markets: int = N_MARKETS, seed: int = SEED) -> dict[str, Any]:
    d = simulate_markets(n_markets, seed=seed)
    ours = {k: float(np.median(getattr(d, k))) for k in PUBLISHED}
    mape = {
        "upp_vs_logit": _mape(d.upp, d.effect_logit),
        "upp_vs_linear": _mape(d.upp, d.effect_linear),
        "foa_vs_logit": _mape(d.foa_logit, d.effect_logit),
        "foa_vs_linear": _mape(d.foa_linear, d.effect_linear),
        "foa_linear_passthrough_vs_logit_truth": _mape(d.foa_linear, d.effect_logit),
        "foa_logit_passthrough_vs_linear_truth": _mape(d.foa_logit, d.effect_linear),
        "logit_simulation_vs_linear": _mape(d.effect_logit, d.effect_linear),
        "linear_simulation_vs_logit": _mape(d.effect_linear, d.effect_logit),
    }
    return {
        "markets": n_markets,
        "seed": seed,
        "redrawn_not_rationalisable": d.redraws,
        "solver_failures": d.solver_failures,
        "medians_ours": ours,
        "medians_published": PUBLISHED,
        "share_delta_hhi_above_200": float(np.mean(d.delta_hhi > 200)),
        "share_delta_hhi_100_to_200": float(np.mean((d.delta_hhi >= 100) & (d.delta_hhi <= 200))),
        "share_delta_hhi_below_100": float(np.mean(d.delta_hhi < 100)),
        "published_share_delta_hhi": {"above_200": 0.65, "100_to_200": 0.15, "below_100": 0.19},
        "mape": mape,
        "mape_published": PUBLISHED_MAPE,
        "correlation_upp_logit_effect": float(np.corrcoef(d.upp, d.effect_logit)[0, 1]),
        "correlation_upp_delta_hhi": float(np.corrcoef(d.upp, d.delta_hhi)[0, 1]),
        "correlation_logit_effect_delta_hhi": float(np.corrcoef(d.effect_logit, d.delta_hhi)[0, 1]),
        "share_upp_overstates_logit_effect": float(np.mean(d.upp > d.effect_logit)),
        "logit_effect_quantiles_5_50_95": np.quantile(d.effect_logit, [0.05, 0.5, 0.95]).tolist(),
        "linear_effect_quantiles_5_50_95": np.quantile(d.effect_linear, [0.05, 0.5, 0.95]).tolist(),
        "_draws": d,
    }


EXHIBIT = 6


def plot(res: dict[str, Any]) -> list[Path]:
    """Exhibit 6: median absolute prediction errors of this reproduction against the paper."""
    m = res["mape"]
    pub = res["mape_published"]
    # (label, error if logit is the truth, error if linear is the truth, ours, in red)
    rows = [
        ("Pricing pressure,\nthis repository", m["upp_vs_logit"], m["upp_vs_linear"], True, True),
        ("Pricing pressure,\npaper", pub["upp_vs_logit"], pub["upp_vs_linear"], False, False),
        (
            "First-order approximation,\nmisspecified pass-through",
            m["foa_linear_passthrough_vs_logit_truth"],
            m["foa_logit_passthrough_vs_linear_truth"],
            True,
            False,
        ),
        (
            "Simulation under the wrong\ndemand form, paper",
            pub["logit_simulation_vs_linear"],
            pub["linear_simulation_vs_logit"],
            False,
            False,
        ),
        (
            "First-order approximation,\ntrue pass-through",
            m["foa_vs_logit"],
            m["foa_vs_linear"],
            True,
            False,
        ),
    ]
    n = len(rows)
    xmax = 2.8

    def build(p: dict[str, str]):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.0, 5.0))
        halo = {"boxstyle": "square,pad=0.12", "fc": p["surface"], "ec": "none"}
        titles = ("If logit demand is the truth", "If linear demand is the truth")
        for k, (ax, title) in enumerate(zip(axes, titles, strict=True)):
            for i, (_, logit_err, linear_err, ours, hot) in enumerate(rows):
                y = n - i
                value = 100 * (logit_err, linear_err)[k]
                ax.plot(
                    [0, value],
                    [y, y],
                    color=p["accent_soft"] if hot else p["rule"],
                    lw=7,
                    solid_capstyle="butt",
                    zorder=2,
                )
                if hot and k == 0:
                    continue
                if ours:
                    dot = {"color": p["accent"] if hot else p["ink_secondary"], "mec": p["surface"]}
                    dot["mew"] = 1.4
                else:  # published values: open dots
                    dot = {"color": p["surface"], "mec": p["ink_secondary"], "mew": 1.5}
                ax.plot(value, y, "o", ms=7, zorder=5, clip_on=False, **dot)
                ax.annotate(
                    f"{value:.2f}" if ours else f"{value:.1f}",
                    (value, y),
                    xytext=(9, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=p["accent"] if hot else p["ink"],
                    fontweight="medium" if hot else "regular",
                )
            ax.set_title(title, loc="left", color=p["muted"], fontsize=9, pad=10)
            ax.set_xlim(0, xmax)
            ax.set_xticks([0, 1, 2])
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.set_xlabel("Percentage points")
        axes[0].annotate(
            "needs the true demand form",
            (0.05, 1),
            xytext=(46, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=p["muted"],
            bbox=halo,
        )
        hot_value = 100 * rows[0][1]
        callout(
            axes[0],
            p,
            (hot_value, n),
            f"{hot_value:.2f}",
            f"points, against {100 * rows[1][1]:.1f}\nreported in the paper",
            dx=26,
            dy=-12,
        )
        axes[0].set_yticks(range(n, 0, -1))
        axes[0].set_yticklabels([r[0] for r in rows], fontsize=9, color=p["ink_secondary"])
        axes[0].set_ylim(0.45, n + 0.6)
        exhibit(
            fig,
            p,
            kicker=f"Exhibit {EXHIBIT} · Reproduction of Miller et al. (2017)",
            title=(
                f"Pricing pressure's median error is {100 * m['upp_vs_logit']:.2f} points under "
                f"logit and {100 * m['upp_vs_linear']:.2f} under linear demand, close to the "
                f"paper's {100 * pub['upp_vs_logit']:.1f} and {100 * pub['upp_vs_linear']:.1f}"
            ),
            subtitle=(
                "Median absolute error of the predicted price effect against the full simulation"
            ),
            source=(
                f"Source: mergerlab, {res['markets']:,} simulated six-firm markets "
                f"(seed {res['seed']}), design of Miller et al. (2017)."
            ),
            wrap=62,
            right_in=0.5,
        )
        fig.subplots_adjust(left=2.3 / 8.0, wspace=0.12)
        return fig

    return save_figure(build, ROOT / "docs" / "figures" / "miller_reproduction")


def public(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if not k.startswith("_")}


if __name__ == "__main__":
    out = run()
    plot(out)
    import json

    print(json.dumps(public(out), indent=1))
