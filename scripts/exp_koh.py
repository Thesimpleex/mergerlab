"""Concentration-based harm for Heinz / Beech-Nut (Koh 2025, Table 1) against full simulation.

Koh's inputs: revenue shares 17.4 / 15.4 / 65 percent, Y = 865 million USD, CES demand with the
elasticity of substitution sigma. Margins follow from sigma (Bertrand), so no margin is assumed.
Table 1 of the paper is reproduced to its printed precision. The first-order formula covers the
merging products with the rivals' prices held fixed, so it is compared first with the exact
compensating variation of that same price change (rivals fixed) and then with the full merger
simulation, in which Gerber re-prices as well; the gap between the two is the rival response.
The margins are those implied by sigma (71% down to 38% for Heinz as sigma rises from 1.5 to
3), not the 25% margin of the illustrative case file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mergerlab import (
    Margins,
    Market,
    Shares,
    calibrate_ces,
    harm_comparison,
    koh_decomposition,
)
from mergerlab.viz import callout, exhibit, save_figure

ROOT = Path(__file__).resolve().parents[1]
SHARES = (0.174, 0.154, 0.65)  # Heinz, Beech-Nut, Gerber
OUTSIDE = 0.022
INCOME = 865.0
SIGMAS = (1.5, 2.0, 2.5, 3.0)
# Koh (2025), Table 1 as printed: sigma, rho1, rho2, delta CS in million USD (absolute value).
PUBLISHED = {
    1.5: {"rho1": 0.37, "rho2": 1.09, "harm": 37.68},
    2.0: {"rho1": 0.59, "rho2": 1.04, "harm": 28.48},
    2.5: {"rho1": 0.73, "rho2": 1.00, "harm": 22.87},
    3.0: {"rho1": 0.84, "rho2": 0.98, "harm": 19.10},
}


def run() -> dict[str, Any]:
    rows = []
    for sigma in SIGMAS:
        margin = 1.0 / (sigma - (sigma - 1.0) * SHARES[0])
        market = Market.build(
            [1.0, 1.0, 1.0],
            Shares.total(SHARES, "revenue", OUTSIDE),
            Margins.lerner([margin, np.nan, np.nan]),
            ["Heinz", "Beech-Nut", "Gerber"],
            revenue=INCOME * (1 - OUTSIDE),
        )
        cal = calibrate_ces(market)
        dec = koh_decomposition(cal.demand, market.prices, cal.costs, 0, 1)
        cmp = harm_comparison(cal.demand, market.prices, cal.costs, 0, 1)
        rows.append(
            {
                "sigma": sigma,
                "phi": dec.phi,
                "v0_musd": dec.v0,
                "rho1": dec.rho1,
                "rho2": dec.rho2,
                "delta_hhi": dec.delta_hhi,
                "harm_first_order_musd": dec.harm,
                "harm_rho2_one_musd": dec.harm_with_rho2_one,
                "harm_exact_rivals_fixed_musd": cmp.exact_rivals_fixed,
                "harm_simulated_musd": cmp.full_simulation,
                "first_order_share_of_exact_rivals_fixed": cmp.first_order / cmp.exact_rivals_fixed,
                "first_order_share_of_simulated": cmp.first_order / cmp.full_simulation,
                "rival_response_musd": cmp.rival_response,
                "rival_response_share_of_simulated": cmp.rival_response_share,
                "curvature_gap_musd": cmp.curvature_gap,
                "gerber_price_change_simulated": float(cmp.price_change_full[2]),
                "implied_heinz_margin": margin,
                "published": PUBLISHED[sigma],
                "abs_diff_to_published_musd": abs(dec.harm - PUBLISHED[sigma]["harm"]),
            }
        )
    return {
        "income_musd": INCOME,
        "shares": dict(zip(["Heinz", "Beech-Nut", "Gerber"], SHARES, strict=True)),
        "outside_share": OUTSIDE,
        "rows": rows,
    }


EXHIBIT = 7


def plot(res: dict[str, Any]) -> list[Path]:
    """Exhibit 7: first-order harm against exact harm with rivals fixed and the full simulation."""
    rows = res["rows"]
    n = len(rows)
    share_lo = min(r["rival_response_share_of_simulated"] for r in rows)
    share_hi = max(r["rival_response_share_of_simulated"] for r in rows)
    xmax = 82.0
    bar = 0.30

    def build(p: dict[str, str]):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        def label(x: float, y: float, text: str, colour: str, *, left: bool, **kw: Any) -> None:
            ax.annotate(
                text,
                (x, y),
                xytext=(6 if left else -6, 0),
                textcoords="offset points",
                ha="left" if left else "right",
                va="center",
                fontsize=kw.pop("fontsize", 8.5),
                color=colour,
                zorder=6,
                **kw,
            )

        for i, r in enumerate(rows):
            y = n - i
            first, exact = r["harm_first_order_musd"], r["harm_exact_rivals_fixed_musd"]
            total, response = r["harm_simulated_musd"], r["rival_response_musd"]
            y1, y2 = y + 0.19, y - 0.19
            ax.barh(y1, first, bar, color=p["ink_secondary"], zorder=3)
            ax.barh(y2, exact, bar, color=p["context"], zorder=3)
            ax.barh(y2, response, bar, left=exact, color=p["accent"], zorder=3)
            label(first, y1, f"{first:.1f}", p["surface"], left=False)
            label(exact, y2, f"{exact:.1f}", p["ink"], left=False)
            label(total, y2, f"{response:.1f}", p["surface"], left=False)
            if i == 0:
                label(0, y1, "First-order formula (Koh 2025)", p["surface"], left=True)
                label(0, y2, "Exact harm, rivals' prices fixed", p["ink"], left=True)
                label(
                    exact,
                    y1,
                    "Gerber's price response",
                    p["accent"],
                    left=True,
                    fontweight="medium",
                )
            label(xmax, y2, f"{total:.1f}", p["ink"], left=False, fontsize=9.5, fontweight="medium")
        ax.annotate(
            "Full simulation",
            (xmax, n + 0.42),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=p["muted"],
        )
        last = rows[-1]
        callout(
            ax,
            p,
            (last["harm_simulated_musd"], 1 - 0.19),
            f"{100 * last['rival_response_share_of_simulated']:.0f}%",
            f"of the full harm at sigma = {last['sigma']:g}\nis Gerber's price response;\n"
            f"{100 * share_lo:.0f}% at sigma = {rows[0]['sigma']:g}",
            dx=40,
            dy=36,
        )
        ax.set_yticks(range(n, 0, -1))
        ax.set_yticklabels(
            [
                f"sigma = {r['sigma']:g}\nHeinz margin {100 * r['implied_heinz_margin']:.0f}%"
                for r in rows
            ],
            fontsize=9,
            color=p["ink_secondary"],
        )
        ax.set_ylim(0.45, n + 0.62)
        ax.set_xlim(0, xmax)
        ax.set_xticks(range(0, 61, 10))
        ax.spines["bottom"].set_bounds(0, 65)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("Annual consumer harm, million USD")
        exhibit(
            fig,
            p,
            kicker=f"Exhibit {EXHIBIT} · Concentration-based harm",
            title=(
                "The first-order formula omits Gerber's price response, which is "
                f"{100 * share_lo:.0f}% to {100 * share_hi:.0f}% of the full simulated harm"
            ),
            subtitle="Heinz / Beech-Nut under CES demand, by elasticity of substitution (sigma)",
            source=(
                "Source: mergerlab; inputs of Koh (2025, Table 1): revenue shares "
                "17.4 / 15.4 / 65%, sales of $865m."
            ),
            wrap=60,
            right_in=0.6,
        )
        fig.subplots_adjust(left=1.5 / 8.0)
        return fig

    return save_figure(build, ROOT / "docs" / "figures" / "koh_first_order")


if __name__ == "__main__":
    import json

    out = run()
    plot(out)
    print(json.dumps(out, indent=1))
