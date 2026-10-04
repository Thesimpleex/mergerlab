"""Run every experiment, write ``docs/results.json`` and redraw every figure in ``docs``.

    python scripts/reproduce_all.py [--screenshots]

No external data are needed: every input is a public figure printed in the case file
(``examples/heinz_beech_nut.toml``) or in the cited papers, and the oracle fixtures are
committed (regenerate them with the scripts in ``scripts/oracles``). Generated images are deleted
before they are redrawn, so no superseded file survives. ``--screenshots`` also regenerates the
banner, the casts, the terminal screenshots, the animation and the HTML memo with
``scripts/make_screenshots.py`` (needs the ``docs`` extra and Google Chrome).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import exp_convention_trap  # noqa: E402
import exp_demand_forms  # noqa: E402
import exp_divestiture  # noqa: E402
import exp_koh  # noqa: E402
import exp_mrrs  # noqa: E402
import exp_oracles  # noqa: E402
import exp_solver_gate  # noqa: E402


def _versions() -> dict[str, str]:
    names = ["mergerlab", "numpy", "scipy", "rich", "matplotlib"]
    return {n: metadata.version(n) for n in names} | {"python": platform.python_version()}


def _clean(directory: Path, *patterns: str) -> None:
    """Delete previously generated images so that renamed or dropped files do not linger."""
    for pattern in patterns:
        for path in directory.glob(pattern):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--screenshots",
        action="store_true",
        help="also regenerate banner, casts, screenshots, animation and memo",
    )
    args = parser.parse_args()
    started = time.time()
    docs = ROOT / "docs"
    _clean(docs / "figures", "*.png")
    if args.screenshots:
        _clean(docs, "banner-*.png")
        _clean(docs / "animations", "*.svg", "*.gif")
        _clean(docs / "screenshots", "*.png")
    results: dict[str, object] = {
        "generated": dt.date.today().isoformat(),
        "versions": _versions(),
        "data": "No external data: inputs are public figures stated in the case file",
    }
    timings: dict[str, float] = {}

    def step(name: str, fn):
        t0 = time.time()
        out = fn()
        timings[name] = round(time.time() - t0, 1)
        print(f"{name}: {timings[name]} s")
        return out

    trap = step("convention_trap", exp_convention_trap.run)
    exp_convention_trap.plot(trap)
    results["convention_trap"] = trap

    forms = step("demand_forms", exp_demand_forms.run)
    exp_demand_forms.plot(forms)
    results["demand_forms"] = forms

    gate = step("solver_gate", exp_solver_gate.run)
    exp_solver_gate.plot(gate)
    results["solver_gate"] = gate

    mrrs = step("miller_reproduction", exp_mrrs.run)
    exp_mrrs.plot(mrrs)
    results["miller_reproduction"] = exp_mrrs.public(mrrs)

    koh = step("koh_first_order", exp_koh.run)
    exp_koh.plot(koh)
    results["koh_first_order"] = koh

    results["divestiture"] = step("divestiture", exp_divestiture.run)
    results["oracles"] = step("oracles", exp_oracles.run)
    results["runtime_seconds"] = timings

    out = ROOT / "docs" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {out} in {time.time() - started:.0f} s")
    if args.screenshots:
        import make_screenshots

        make_screenshots.main()
        print(f"everything regenerated in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
