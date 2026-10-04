"""The convention trap: revenue diversion fed into Werden's quantity formula.

Epstein-Rubinfeld three-firm PCAIDS example (revenue shares 20/30/50, prices 2.9/3.4/2.2,
own elasticity -3, market elasticity -1); firms A and B merge. The AIDS-type revenue
diversion ratio (A to B: 0.375) differs from the quantity diversion ratio (0.213). Plugging
the former into the quantity-based CMCR formula doubles the answer for A.

R's AIDS diversion is a ratio of budget-share slopes. It coincides with the revenue-change
ratio only at a market elasticity of -1; the second part of ``run`` repeats the example at -1.5
to show that converting it as a revenue ratio then gives a wrong CMCR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mergerlab import (
    Diversion,
    DiversionBasis,
    Margins,
    Market,
    Shares,
    calibrate_pcaids,
    cmcr_from_demand,
    cmcr_from_diversion,
    cmcr_two_product,
    concentration,
    diversion_from_demand,
)
from mergerlab.viz import callout, exhibit, save_figure

ROOT = Path(__file__).resolve().parents[1]


def run() -> dict[str, Any]:
    p = np.array([2.9, 3.4, 2.2])
    market = Market.build(
        p, Shares.within([0.2, 0.3, 0.5], "revenue"), Margins.lerner([np.nan] * 3), list("ABC")
    )
    cal = calibrate_pcaids(market, own_elasticity=-3.0, market_elasticity=-1.0)
    post = market.ownership.merged(["A", "B"])
    margins = cal.fitted_margins
    conc = concentration(market.shares, market.ownership, post)
    revenue_div = diversion_from_demand(cal.demand, p, DiversionBasis.REVENUE)
    quantity_div = diversion_from_demand(cal.demand, p, DiversionBasis.QUANTITY)
    own_elasticity = np.diag(cal.demand.elasticities(p))
    model = cmcr_from_demand(cal.demand, p, cal.costs, post)
    typed = cmcr_from_diversion(p, Margins.lerner(margins), quantity_div, market.ownership, post)
    r_ab, r_ba = revenue_div.matrix[0, 1], revenue_div.matrix[1, 0]
    wrong_a = cmcr_two_product(margins[0], margins[1], r_ab, r_ba, p[1] / p[0])
    wrong_b = cmcr_two_product(margins[1], margins[0], r_ba, r_ab, p[0] / p[1])
    refused = ""
    try:
        cmcr_from_diversion(p, Margins.lerner(margins), revenue_div, market.ownership, post)
    except ValueError as exc:
        refused = str(exc)
    roundtrip = revenue_div.to_quantity(p, own_elasticity)
    off_unit = _market_elasticity_minus_one_and_a_half(p, market)
    return {
        "prices": p.tolist(),
        "lerner_margins": margins.tolist(),
        "own_elasticities": own_elasticity.tolist(),
        "revenue_diversion_ab": float(r_ab),
        "revenue_diversion_ba": float(r_ba),
        "quantity_diversion_ab": float(quantity_div.matrix[0, 1]),
        "quantity_diversion_ba": float(quantity_div.matrix[1, 0]),
        "cmcr_correct_a": float(model.relative[0]),
        "cmcr_correct_b": float(model.relative[1]),
        "cmcr_from_typed_quantity_diversion_a": float(typed.relative[0]),
        "cmcr_wrong_a": float(wrong_a),
        "cmcr_wrong_b": float(wrong_b),
        "conversion_roundtrip_error": float(np.max(np.abs(roundtrip.matrix - quantity_div.matrix))),
        "typed_api_error_message": refused,
        "hhi_pre": conc.hhi_pre,
        "hhi_post": conc.hhi_post,
        "market_elasticity_minus_1_5": off_unit,
    }


def _market_elasticity_minus_one_and_a_half(p: np.ndarray, market: Market) -> dict[str, Any]:
    """The same example at market elasticity -1.5, where share and revenue diversions differ."""
    eps_m = -1.5
    cal = calibrate_pcaids(market, own_elasticity=-3.0, market_elasticity=eps_m)
    post = market.ownership.merged(["A", "B"])
    margins = Margins.lerner(cal.fitted_margins)
    own = np.diag(cal.demand.elasticities(p))
    w = market.shares.values
    r_style = diversion_from_demand(cal.demand, p, DiversionBasis.SHARE)
    true_q = diversion_from_demand(cal.demand, p, DiversionBasis.QUANTITY)
    via_share = r_style.to_quantity(p, own, shares=w, market_elasticity=eps_m)
    via_revenue = Diversion.revenue(r_style.matrix).to_quantity(p, own)
    true_cmcr = cmcr_from_demand(cal.demand, p, cal.costs, post)
    cm_share = cmcr_from_diversion(p, margins, via_share, market.ownership, post)
    cm_revenue = cmcr_from_diversion(p, margins, via_revenue, market.ownership, post)
    return {
        "market_elasticity": eps_m,
        "r_style_share_diversion_ab": float(r_style.matrix[0, 1]),
        "revenue_convention_ab": float(
            true_q.to_revenue(p, own).matrix[0, 1],
        ),
        "quantity_diversion_ab_true": float(true_q.matrix[0, 1]),
        "quantity_diversion_ab_via_share_conversion": float(via_share.matrix[0, 1]),
        "quantity_diversion_ab_via_revenue_conversion": float(via_revenue.matrix[0, 1]),
        "cmcr_true": [float(x) for x in true_cmcr.relative[:2]],
        "cmcr_via_share_conversion": [float(x) for x in cm_share.relative[:2]],
        "cmcr_via_revenue_conversion": [float(x) for x in cm_revenue.relative[:2]],
    }


EXHIBIT = 3


def plot(res: dict[str, Any]) -> list[Path]:
    """Exhibit 3: CMCR with the correct quantity diversion and with revenue diversion plugged in."""
    correct = [100 * res["cmcr_correct_a"], 100 * res["cmcr_correct_b"]]
    wrong = [100 * res["cmcr_wrong_a"], 100 * res["cmcr_wrong_b"]]
    w = 0.30
    gap = 0.02
    xs = np.arange(2)

    def build(p: dict[str, str]):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        right_bars = ax.bar(xs - w / 2 - gap, correct, w, color=p["ink_secondary"], zorder=3)
        wrong_bars = ax.bar(xs + w / 2 + gap, wrong, w, color=p["accent"], zorder=3)
        for bars, text in ((right_bars, "Quantity\ndiversion"), (wrong_bars, "Revenue\ndiversion")):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.2,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=p["surface"],
                    linespacing=1.3,
                    zorder=4,
                )
        for bar in (right_bars[0], right_bars[1], wrong_bars[1]):
            hot = bar in wrong_bars
            ax.annotate(
                f"{bar.get_height():.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="medium" if hot else "regular",
                color=p["accent"] if hot else p["ink"],
            )
        top = wrong_bars[0]
        callout(
            ax,
            p,
            (top.get_x() + top.get_width(), top.get_height()),
            f"{wrong[0]:.1f}%",
            f"required for A; with the quantity\ndiversion ratio it is {correct[0]:.1f}%",
            dx=26,
            dy=-4,
        )
        ax.set_xticks(xs, ["Product A", "Product B"], fontsize=9.5)
        ax.tick_params(axis="x", length=0, pad=8, labelcolor=p["ink_secondary"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 40)
        ax.set_yticks([0, 20, 40])
        ax.set_yticklabels(["0", "20", "40%"])
        ax.set_xlabel("Epstein-Rubinfeld three-firm example; A and B merge")
        exhibit(
            fig,
            p,
            kicker=f"Exhibit {EXHIBIT} · Convention trap",
            title=(
                "A revenue diversion ratio fed into the quantity formula nearly doubles the "
                "synergy required for product A"
            ),
            subtitle=(
                "Compensating marginal cost reduction (CMCR) of the merging products, "
                "% of marginal cost"
            ),
            source=(
                "Source: mergerlab; Epstein and Rubinfeld (2002) example, prices 2.9 / 3.4 / 2.2, "
                "revenue shares 20 / 30 / 50%."
            ),
            wrap=60,
            right_in=0.6,
        )
        fig.subplots_adjust(left=1.0 / 8.0)
        return fig

    return save_figure(build, ROOT / "docs" / "figures" / "convention_trap")


if __name__ == "__main__":
    out = run()
    plot(out)
    print(out)
