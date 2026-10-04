"""Regenerate the banner, casts, screenshots and animation in ``docs/`` from real program output.

    python scripts/make_screenshots.py

Needs the ``docs`` extra (matplotlib, playwright), Google Chrome and ``docs/results.json``
(written by ``scripts/reproduce_all.py``, which calls this script with ``--screenshots``).
Writes

* ``docs/banner-{light,dark}.png``: README banner; the key number comes from ``results.json``,
* ``docs/animations/simulate-{light,dark}.svg``: animated terminal cast of ``mergerlab simulate``,
* ``docs/animations/solver_iteration-{light,dark}.gif``: Exhibit 4, the textbook iteration
  against the Morrow-Skerlos fixed point on the pyblp nested-logit scenario,
* ``docs/screenshots/{solver_gate,screen}-{light,dark}.png``: terminal screenshots,
* ``docs/screenshots/memo-*-light.png`` and ``docs/examples/heinz_beech_nut_memo.html``: the
  merger memo and two crops of it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from animate import save_animation, terminal_cast_svg  # noqa: E402
from banner import render_banner  # noqa: E402
from exp_demand_forms import banner_chart  # noqa: E402
from exp_solver_gate import EXHIBIT_ITERATION, iteration_trace  # noqa: E402
from screenshots import report_screenshot, terminal_screenshot  # noqa: E402

from mergerlab.viz import callout, end_label, exhibit, style  # noqa: E402

CASE = "examples/heinz_beech_nut.toml"
MEMO = ROOT / "docs" / "examples" / "heinz_beech_nut_memo.html"
RESULTS = ROOT / "docs" / "results.json"
COLUMNS = 122
PYTHON = sys.executable
THEMES = ("light", "dark")
WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def _cli(*args: str) -> list[str]:
    return [PYTHON, "-m", "mergerlab.cli", *args]


def build_memo() -> None:
    MEMO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        _cli("simulate", CASE, "--report", str(MEMO)), cwd=ROOT, check=True, capture_output=True
    )


def banner() -> None:
    """README banner: the spread of predicted price effects across the demand forms."""
    res = json.loads(RESULTS.read_text())["demand_forms"]
    lo, hi = res["point_range"]
    n_forms = sum(1 for f in res["forms"].values() if f["status"] == "ok")
    with tempfile.TemporaryDirectory() as tmp:
        light, dark = banner_chart(res, Path(tmp) / "chart")
        render_banner(
            ROOT / "docs" / "banner",
            name="mergerlab",
            claim="Calibrated merger simulation and antitrust screening from shares, prices "
            "and margins.",
            kicker="Open-source Python library · Merger screening",
            value=f"{100 * lo:.1f}% to {100 * hi:.1f}%",
            label=(
                "predicted price increase of the merging firms across "
                f"{WORDS.get(n_forms, n_forms)} demand forms calibrated to the same prices, "
                "shares and one margin (illustrative Heinz / Beech-Nut case)."
            ),
            chart_light=light,
            chart_dark=dark,
            footer_left="github.com/Thesimpleex/mergerlab",
            footer_right="Python 3.11+ · MIT licence",
        )


def casts() -> None:
    for theme in THEMES:
        terminal_cast_svg(
            _cli("simulate", CASE),
            ROOT / "docs" / "animations" / f"simulate-{theme}.svg",
            title="mergerlab",
            display_command=f"mergerlab simulate {CASE}",
            columns=COLUMNS,
            cwd=ROOT,
            theme=theme,
        )


def terminal_shots() -> None:
    out = ROOT / "docs" / "screenshots"
    for theme in THEMES:
        terminal_screenshot(
            [PYTHON, "scripts/exp_solver_gate.py", "--table", "--results", str(RESULTS)],
            out / f"solver_gate-{theme}.png",
            title="solver gate",
            display_command="python scripts/exp_solver_gate.py --table",
            theme=theme,
            columns=118,
            cwd=ROOT,
        )
        terminal_screenshot(
            _cli("screen", "--shares", "65", "17.4", "15.4", "--merge", "2", "3"),
            out / f"screen-{theme}.png",
            title="mergerlab",
            display_command="mergerlab screen --shares 65 17.4 15.4 --merge 2 3",
            theme=theme,
            columns=100,
            cwd=ROOT,
        )


def memo_shots() -> None:
    """Two crops of the memo: key figures with the findings, and Exhibit 1 with its table.

    The crops are taken from a temporary copy with page-style padding, so that text does not touch
    the edge of the image; the memo itself is not changed.
    """
    out = ROOT / "docs" / "screenshots"
    pad = (
        "#front,#price-block{padding:44px 56px;background:#fff}"
        "#price-block>:first-child{margin-top:0}"
    )
    html = MEMO.read_text(encoding="utf-8").replace("</head>", f"<style>{pad}</style></head>", 1)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "memo.html"
        page.write_text(html, encoding="utf-8")
        report_screenshot(page, out / "memo-summary-light.png", width=1100, selector="#front")
        report_screenshot(page, out / "memo-effects-light.png", width=1100, selector="#price-block")


def solver_animation() -> None:
    """Exhibit 4: lines grow iteration by iteration; the callout appears on the last frame."""
    tr = iteration_trace()
    naive = 100 * np.array(tr["naive_max_increase"])
    zeta = 100 * np.array(tr["zeta_max_increase"])
    target = 100 * tr["pyblp_max_increase"]
    n = len(naive)

    def draw(ax, p, i):
        x = np.arange(i + 1)
        ax.axhline(target, color=p["context_strong"], lw=1.0, zorder=1)
        ax.plot(x, zeta[: i + 1], color=p["ink_secondary"], lw=1.9, zorder=3)
        ax.plot(x, naive[: i + 1], color=p["accent"], lw=2.3, zorder=4)
        dot = {"ms": 6.5, "mec": p["surface"], "mew": 1.3, "clip_on": False}
        ax.plot(i, zeta[i], "o", color=p["ink_secondary"], zorder=5, **dot)
        if i < n - 1:
            ax.plot(i, naive[i], "o", color=p["accent"], zorder=6, **dot)
        ax.set_xlim(0, n - 1)
        ax.set_ylim(0, 205)
        ax.set_xticks(range(0, n, 10))
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_yticklabels(["0", "50", "100", "150", "200%"])
        ax.set_xlabel("Iteration")
        ax.grid(axis="x", visible=False)
        end_label(ax, n - 1, 168, "Textbook iteration", p["accent"], bold=True)
        end_label(
            ax,
            n - 1,
            target,
            f"Morrow-Skerlos fixed point\nconverges to +{target:.1f}%",
            p["ink_secondary"],
        )
        ax.text(
            0.012,
            0.965,
            f"Iteration {i}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=p["muted"],
        )
        if i == n - 1:
            callout(
                ax,
                p,
                (i, naive[i]),
                f"+{naive[i]:.1f}%",
                f"alternating with +{naive[i - 1]:.1f}%;\nneither is an equilibrium",
                dx=30,
                dy=-40,
                ha="right",
            )
        if i == 0:
            exhibit(
                ax.figure,
                p,
                kicker=f"Exhibit {EXHIBIT_ITERATION} · Solver iteration",
                title=(
                    "On the same equations and start, the textbook iteration cycles while the "
                    "Morrow-Skerlos fixed point converges"
                ),
                subtitle="Largest post-merger price increase after each iteration, %",
                source=(
                    "Source: pyblp nested-logit scenario with nesting parameter 0.6; both "
                    "iterations start at marginal cost plus one."
                ),
                wrap=58,
                right_in=2.0,
            )

    save_animation(
        draw,
        n,
        ROOT / "docs" / "animations" / "solver_iteration",
        style=style,
        figsize=(8.0, 5.0),
        fps=12,
        dpi=100,
        hold_last=36,
    )


def main() -> None:
    (ROOT / "docs" / "animations").mkdir(parents=True, exist_ok=True)
    build_memo()
    banner()
    casts()
    terminal_shots()
    memo_shots()
    solver_animation()
    print("banner, casts, screenshots and animation written to docs/")


if __name__ == "__main__":
    main()
