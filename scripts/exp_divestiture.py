"""Single-product divestiture ranking in a stylised two-brand-by-two-brand merger.

Firm A sells brands a1 and a2, firm B sells b1 and b2, a fringe brand c is independent. All brands
are priced near one; shares are quantity shares of a market with a 20 percent outside good; the
Lerner margin of a1 (35 percent) is the only observed margin. Each candidate divestiture moves one
brand of the merging parties to an independent owner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mergerlab import Margins, Market, ModelSpec, Shares, evaluate_divestitures


def run() -> dict[str, Any]:
    market = Market.build(
        [1.0, 1.1, 0.9, 1.0, 1.0],
        Shares.total([0.18, 0.08, 0.20, 0.06, 0.28], "quantity"),
        Margins.lerner([0.35, np.nan, np.nan, np.nan, np.nan]),
        ["A", "A", "B", "B", "C"],
        labels=["a1", "a2", "b1", "b2", "c"],
        revenue=1000.0,
    )
    post = market.ownership.merged(["A", "B"])
    out: dict[str, Any] = {"revenue_musd": 1000.0, "models": {}}
    for kind in ("logit", "ces"):
        spec = ModelSpec(kind)
        if kind == "ces":
            market_ces = Market.build(
                market.prices,
                Shares.total(market.revenue_shares_within() * 0.8, "revenue"),
                Margins.lerner(market.lerner),
                ["A", "A", "B", "B", "C"],
                labels=list(market.labels),
                revenue=1000.0,
            )
            report = evaluate_divestitures(spec, market_ces, post)
        else:
            report = evaluate_divestitures(spec, market, post)
        base = report.baseline
        out["models"][kind] = {
            "baseline_harm_musd": base.consumer_harm,
            "baseline_max_price_change": float(np.max(base.price_change)),
            "options": [
                {
                    "divested": o.label,
                    "remaining_harm_musd": o.consumer_harm,
                    "remaining_share_of_baseline": o.consumer_harm / base.consumer_harm,
                    "max_price_change": o.max_price_change,
                }
                for o in report.options
            ],
        }
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=1))
