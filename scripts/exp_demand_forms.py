"""One deal, several demand curves: Heinz / Beech-Nut calibrated five ways.

Runs ``examples/heinz_beech_nut.toml`` (illustrative research on public shares and stated
assumptions), writes the headline numbers and the figure ``docs/figures/demand_forms``.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mergerlab import analyze, load_case
from mergerlab.summary import summarize
from mergerlab.viz import callout, exhibit, save_figure

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "heinz_beech_nut.toml"


def run() -> dict[str, Any]:
    case = load_case(CASE)
    an = analyze(case)
    s = summarize(an)
    sim = an.simulation
    mc = an.monte_carlo
    assert mc is not None
    forms: dict[str, Any] = {}
    outcomes = {o.spec.kind: o for o in sim.outcomes}
    for r in s.rows:
        entry: dict[str, Any] = {"label": r.label, "status": r.status}
        if r.status == "ok":
            entry.update(
                price_change_heinz=r.party_changes[0],
                price_change_beech_nut=r.party_changes[1],
                price_change_average=r.average_change,
                first_order_average=r.first_order_average,
                band_p05=r.band[0] if r.band else None,
                band_p50=r.band[1] if r.band else None,
                band_p95=r.band[2] if r.band else None,
                cmcr_average=r.cmcr_average,
                cmcr_p05=r.cmcr_band[0] if r.cmcr_band else None,
                cmcr_p50=r.cmcr_band[1] if r.cmcr_band else None,
                cmcr_p95=r.cmcr_band[2] if r.cmcr_band else None,
                synergies_musd=r.synergy_total,
                consumer_harm_musd=r.consumer_harm,
                consumer_harm_share=r.consumer_harm_share,
                gerber_price_change=r.outsider_changes[0] if r.outsider_changes else None,
                foc_residual=r.residual,
                parameters=r.parameters,
                implied_margins=dict(
                    zip(
                        case.market.labels,
                        outcomes[r.kind].calibration.fitted_margins.tolist(),
                        strict=True,
                    )
                ),
            )
        mc_counts = mc.counts(r.kind)
        entry["monte_carlo"] = mc_counts
        entry["monte_carlo_successful_draws"] = mc_counts["ok"]
        sigma_cover = mc.covered_range(r.kind, "sigma") if r.kind == "nested_logit" else None
        if sigma_cover is not None:
            sig = mc.parameter_draws["sigma"]
            ok = np.array([st == "ok" for st in mc.status[r.kind]])
            edges = np.linspace(mc.priors.sigma_range[0], mc.priors.sigma_range[1], 5)
            entry["nesting_parameter_prior_range"] = list(mc.priors.sigma_range)
            entry["nesting_parameter_range_of_successful_draws"] = list(sigma_cover)
            entry["nesting_parameter_mean_successful"] = float(sig[ok].mean())
            entry["nesting_parameter_mean_failed"] = float(sig[~ok].mean())
            entry["failure_share_by_nesting_parameter_quartile"] = [
                float(1.0 - ok[(sig >= lo) & (sig <= hi)].mean()) for lo, hi in pairwise(edges)
            ]
        entry["monte_carlo_failure_share"] = mc.failure_share(r.kind)
        if r.hm is not None:
            entry["hm_passes"] = r.hm.passes
            entry["hm_max_increase"] = r.hm.max_increase
        forms[r.kind] = entry
    ens = s.ensemble_band
    c = sim.concentration
    return {
        "case": case.name,
        "draws": mc.draws,
        "seed": mc.seed,
        "hhi_pre": c.hhi_pre,
        "hhi_post": c.hhi_post,
        "delta_hhi": c.delta_hhi,
        "merged_share": c.merged_share,
        "screens": {r.ruleset: r.status for r in sim.screens},
        "forms": forms,
        "point_range": list(s.point_range) if s.point_range else None,
        "cmcr_range": list(s.cmcr_range) if s.cmcr_range else None,
        "synergy_range_musd": list(s.synergy_range) if s.synergy_range else None,
        "harm_range_musd": list(s.harm_range) if s.harm_range else None,
        "ensemble_p05_p50_p95": list(ens) if ens else None,
        "failure_share_overall": mc.failure_share(),
        "hm_candidate": [case.market.labels[i] for i in (case.hm_products or ())],
    }


EXHIBIT_PRICE = 1
EXHIBIT_SYNERGY = 2
HIGHLIGHT = "nested_logit"
FIGURES = ROOT / "docs" / "figures"


def _ranked(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Calibrated forms, largest predicted price increase first (same order in every exhibit)."""
    rows = [{"kind": k, **f} for k, f in res["forms"].items() if f["status"] == "ok"]
    return sorted(rows, key=lambda f: -f["price_change_average"])


def draw_forms(
    ax: Axes,
    p: dict[str, str],
    rows: list[dict[str, Any]],
    *,
    point: str,
    band: str,
    xmax: float,
    callout_row: bool = True,
    band_width: float = 9.0,
    dot_size: float = 7.5,
    label_size: float = 9.5,
) -> None:
    """Dot and 5th to 95th percentile bar per demand form, the highlighted form in red.

    With ``callout_row`` the highlighted form is left without dot and value label, because
    ``viz.callout`` draws them.
    """
    n = len(rows)
    for i, f in enumerate(rows):
        y = n - i
        hot = f["kind"] == HIGHLIGHT
        lo, hi = 100 * f[f"{band}_p05"], 100 * f[f"{band}_p95"]
        value = 100 * f[point]
        ax.plot(
            [lo, hi],
            [y, y],
            color=p["accent_soft"] if hot else p["rule"],
            lw=band_width,
            solid_capstyle="round",
            zorder=2,
        )
        if hot and callout_row:
            continue
        ax.plot(
            value,
            y,
            "o",
            ms=dot_size,
            color=p["accent"] if hot else p["ink_secondary"],
            mec=p["surface"],
            mew=1.4,
            zorder=5,
        )
        ax.annotate(
            f"{value:.1f}%",
            (value, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=p["accent"] if hot else p["ink"],
            fontweight="medium" if hot else "regular",
        )
    ax.set_yticks(range(n, 0, -1))
    ax.set_yticklabels([f["label"] for f in rows], fontsize=label_size, color=p["ink_secondary"])
    for tick, f in zip(ax.get_yticklabels(), rows, strict=True):
        if f["kind"] == HIGHLIGHT:
            tick.set_color(p["accent"])
            tick.set_fontweight("medium")
    ax.set_ylim(0.45, n + 0.55)
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def _key(ax: Axes, p: dict[str, str]) -> None:
    ax.text(
        1.0,
        0.04,
        "Dot: point estimate\nBar: 5th to 95th percentile of the Monte Carlo draws",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=p["muted"],
        linespacing=1.5,
    )


def _source(res: dict[str, Any], extra: str = "") -> str:
    return (
        "Source: mergerlab simulation, Heinz / Beech-Nut case file (illustrative inputs); "
        f"{extra}{res['draws']:,} Monte Carlo draws per form, seed {res['seed']}."
    )


def plot(res: dict[str, Any]) -> list[Path]:
    """Exhibits 1 (price effects) and 2 (synergies needed to keep prices flat)."""
    rows = _ranked(res)
    hot = next(f for f in rows if f["kind"] == HIGHLIGHT)
    others = [f for f in rows if f["kind"] != HIGHLIGHT]
    y_hot = len(rows)  # the highlighted form is ranked first
    lo, hi = res["point_range"]
    o_lo = 100 * min(f["price_change_average"] for f in others)
    o_hi = 100 * max(f["price_change_average"] for f in others)
    c_lo, c_hi = res["cmcr_range"]
    cheapest = min(rows, key=lambda f: f["cmcr_average"])

    def frame(fig: Any) -> None:
        fig.subplots_adjust(left=1.2 / 8.0)

    def build_price(p: dict[str, str]):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        draw_forms(ax, p, rows, point="price_change_average", band="band", xmax=25)
        ax.set_xlabel("Price increase of the merging firms, % (revenue-weighted average)")
        callout(
            ax,
            p,
            (100 * hot["price_change_average"], y_hot),
            f"{100 * hot['price_change_average']:.1f}%",
            f"Nested logit; the other four forms\npredict {o_lo:.1f}% to {o_hi:.1f}%",
            dx=30,
            dy=-34,
        )
        _key(ax, p)
        exhibit(
            fig,
            p,
            kicker=f"Exhibit {EXHIBIT_PRICE} · Price effects",
            title=(
                f"One deal, five demand forms: predicted price increases range from "
                f"{100 * lo:.1f}% to {100 * hi:.1f}%"
            ),
            subtitle="Average price increase of Heinz and Beech-Nut by demand form, %",
            source=_source(res),
            wrap=62,
            right_in=0.6,
        )
        frame(fig)
        return fig

    def build_synergy(p: dict[str, str]):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        draw_forms(ax, p, rows, point="cmcr_average", band="cmcr", xmax=35)
        ax.set_xlabel("Compensating marginal cost reduction, % of marginal cost")
        callout(
            ax,
            p,
            (100 * hot["cmcr_average"], y_hot),
            f"{100 * hot['cmcr_average']:.1f}%",
            f"of marginal cost, or ${hot['synergies_musd']:.1f}m a year, under nested logit;\n"
            f"{cheapest['label']} needs {100 * cheapest['cmcr_average']:.1f}% "
            f"(${cheapest['synergies_musd']:.1f}m)",
            dx=30,
            dy=-34,
        )
        _key(ax, p)
        exhibit(
            fig,
            p,
            kicker=f"Exhibit {EXHIBIT_SYNERGY} · Synergies",
            title=(
                f"Keeping prices flat takes marginal-cost savings of {100 * c_lo:.1f}% to "
                f"{100 * c_hi:.1f}%, depending on the demand form"
            ),
            subtitle=(
                "Compensating marginal cost reduction (CMCR) of Heinz and Beech-Nut, "
                "% of marginal cost"
            ),
            source=_source(res, "CMCR after Werden (1996); "),
            wrap=62,
            right_in=0.6,
        )
        frame(fig)
        return fig

    return [
        *save_figure(build_price, FIGURES / "demand_forms"),
        *save_figure(build_synergy, FIGURES / "synergies"),
    ]


def banner_chart(res: dict[str, Any], stem: str | Path) -> list[Path]:
    """Minimal version of Exhibit 1 for the README banner (no exhibit text)."""
    rows = _ranked(res)

    def build(p: dict[str, str]):
        fig, ax = plt.subplots(figsize=(5.6, 4.3))
        draw_forms(
            ax,
            p,
            rows,
            point="price_change_average",
            band="band",
            xmax=25,
            callout_row=False,
            band_width=7.5,
            dot_size=6.5,
            label_size=9,
        )
        ax.set_title(
            "Average price increase of Heinz and Beech-Nut, by demand form",
            loc="left",
            color=p["muted"],
            fontsize=9,
            pad=14,
        )
        fig.subplots_adjust(left=0.2, right=0.97, top=0.88, bottom=0.1)
        return fig

    return save_figure(build, stem)


if __name__ == "__main__":
    out = run()
    plot(out)
    print(f"point range {out['point_range']}, ensemble {out['ensemble_p05_p50_p95']}")
