"""Agreement with the reference implementations, measured on the committed fixtures.

The fixtures in ``tests/data`` come from pyblp and the CRAN package ``antitrust`` (see
``scripts/oracles``). This script re-solves every case with mergerlab and records the largest
deviations; the same cases are asserted in the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mergerlab import (
    Diversion,
    Logit,
    Margins,
    Market,
    NestedLogit,
    Ownership,
    Shares,
    calibrate_logit,
    calibrate_pcaids,
    cmcr_from_demand,
    cmcr_from_diversion,
    cmcr_two_product,
    solve_bertrand,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


def _pyblp(fix: dict[str, Any]) -> dict[str, Any]:
    out = []
    for sc in fix["scenarios"]:
        delta = np.array(sc["delta"])
        demand = (
            Logit(delta, sc["alpha"])
            if sc["sigma"] is None
            else NestedLogit(delta, sc["alpha"], sc["sigma"], sc["nests"])
        )
        costs = np.array(sc["costs"])
        pre = Ownership.from_owners(sc["owners_pre"])
        post = Ownership.from_owners(sc["owners_post"])
        eq0 = solve_bertrand(demand, costs, pre, costs + 1.0)
        eq1 = solve_bertrand(demand, costs, post, eq0.prices)
        ref0, ref1 = np.array(sc["pre"]["prices"]), np.array(sc["post"]["prices"])
        out.append(
            {
                "name": sc["name"],
                "max_rel_error_pre": float(np.max(np.abs(eq0.prices / ref0 - 1))),
                "max_rel_error_post": float(np.max(np.abs(eq1.prices / ref1 - 1))),
                "largest_price_effect": float(np.max(ref1 / ref0 - 1)),
                "residual": eq1.residual,
            }
        )
    return {
        "pyblp_version": fix["pyblp_version"],
        "scenarios": out,
        "max_rel_error": max(max(s["max_rel_error_pre"], s["max_rel_error_post"]) for s in out),
    }


def _r(fix: dict[str, Any]) -> dict[str, Any]:
    f = fix["pcaids_epstein_rubinfeld"]
    p = np.array(f["prices"])
    mk = Market.build(
        p, Shares.within(f["revenue_shares"], "revenue"), Margins.lerner([np.nan] * 3), list("ABC")
    )
    cal = calibrate_pcaids(mk, f["own_elasticity_first"], f["market_elasticity"])
    post = mk.ownership.merged(["A", "B"])
    eq = solve_bertrand(cal.demand, cal.costs, post, p)
    cm = cmcr_from_demand(cal.demand, p, cal.costs, post)
    er = {
        "price_change_pct": (100 * (eq.prices / p - 1)).tolist(),
        "price_change_pct_r": f["price_change_pct"],
        "cmcr_pct": (100 * cm.relative[:2]).tolist(),
        "cmcr_pct_r": f["cmcr_pct"],
    }
    pe = fix["pcaids_market_elasticity"]
    pe_err = 0.0
    share_err = 0.0
    for case in pe["cases"]:
        pp = np.array(pe["prices"])
        ww = np.array(pe["revenue_shares"])
        pmk = Market.build(
            pp, Shares.within(ww, "revenue"), Margins.lerner([np.nan] * 3), list("ABC")
        )
        pcal = calibrate_pcaids(pmk, pe["own_elasticity_first"], case["market_elasticity"])
        ppost = pmk.ownership.merged(["A", "B"])
        peq = solve_bertrand(pcal.demand, pcal.costs, ppost, pp)
        pcm = cmcr_from_demand(pcal.demand, pp, pcal.costs, ppost)
        pe_err = max(
            pe_err,
            float(np.max(np.abs(100 * (peq.prices / pp - 1) - case["price_change_pct"]))),
            float(np.max(np.abs(100 * pcm.relative[:2] - case["cmcr_pct"]))),
        )
        own = np.diag(np.array(case["elasticities"]))
        q = Diversion.share(np.array(case["share_diversion"])).to_quantity(
            pp, own, shares=ww, market_elasticity=case["market_elasticity"]
        )
        via = cmcr_from_diversion(pp, Margins.lerner(case["margins"]), q, pmk.ownership, ppost)
        share_err = max(share_err, float(np.max(np.abs(100 * via.relative[:2] - case["cmcr_pct"]))))
    w = fix["cmcr_werden_table1"]
    grid = np.array(w["cmcr_pct"])
    dev = max(
        abs(100 * cmcr_two_product(m, m, d, d, 1.0) - grid[i, j])
        for i, d in enumerate(w["diversions"])
        for j, m in enumerate(w["margins"])
    )
    doc = fix["cmcr_bertrand_doc"]
    res = cmcr_from_diversion(
        np.array(doc["prices"]),
        Margins.lerner(doc["margins"]),
        Diversion.quantity(doc["diversion"]),
        Ownership.from_owners(["M", "N", "N"]),
        Ownership(np.ones((3, 3))),
    )
    lg = fix["logit_known_outside"]
    lp = np.array(lg["prices"])
    lm = Market.build(
        lp,
        Shares.total(lg["quantity_shares"], "quantity"),
        Margins.lerner([np.nan if v is None else v for v in lg["margins"]]),
        lg["owner_pre"],
    )
    lcal = calibrate_logit(lm)
    lpost = lm.ownership.merged(["A", "C"])
    leq = solve_bertrand(lcal.demand, lcal.costs, lpost, lp)
    return {
        "antitrust_version": fix["antitrust_version"],
        "r_version": fix["r_version"],
        "epstein_rubinfeld": er,
        "epstein_rubinfeld_max_abs_error_pct_points": float(
            max(
                np.max(np.abs(np.array(er["price_change_pct"]) - er["price_change_pct_r"])),
                np.max(np.abs(np.array(er["cmcr_pct"]) - er["cmcr_pct_r"])),
            )
        ),
        "pcaids_market_elasticities": [c["market_elasticity"] for c in pe["cases"]],
        "pcaids_market_elasticity_max_abs_error_pct_points": pe_err,
        "share_diversion_cmcr_max_abs_error_pct_points": share_err,
        "werden_table1_cells": int(grid.size),
        "werden_table1_range_pct": [float(grid.min()), float(grid.max())],
        "werden_table1_max_abs_error_pct_points": float(dev),
        "cmcr_bertrand_doc_max_abs_error_pct_points": float(
            np.max(np.abs(100 * res.relative - doc["cmcr_pct"]))
        ),
        "logit_alpha": float(lcal.parameters["alpha"]),
        "logit_alpha_r": lg["alpha"],
        "logit_price_change_max_abs_error_pct_points": float(
            np.max(np.abs(100 * (leq.prices / lp - 1) - lg["price_change_pct"]))
        ),
    }


def run() -> dict[str, Any]:
    pyblp_fix = json.loads((DATA / "pyblp_merger.json").read_text())
    r_fix = json.loads((DATA / "r_antitrust.json").read_text())
    return {"pyblp": _pyblp(pyblp_fix), "antitrust": _r(r_fix)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
