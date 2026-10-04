"""Concentration measures and versioned guideline screens.

Thresholds are taken from the official texts:

* U.S. Department of Justice and Federal Trade Commission, *Merger Guidelines* (December
  18, 2023), section 2.1 (Guideline 1), pp. 5-6.
* U.S. Department of Justice and Federal Trade Commission, *Horizontal Merger
  Guidelines* (August 19, 2010), section 5.3, pp. 18-19.
* European Commission, *Guidelines on the assessment of horizontal mergers under the
  Council Regulation on the control of concentrations between undertakings*, OJ C 31,
  5.2.2004, p. 5, paragraphs 17 to 21.

A screen is a first indicator. None of the guidelines turns these numbers into a
finding; the U.S. 2023 presumption is rebuttable and the EU levels "do not give rise to a
presumption of either the existence or the absence" of concerns (paragraph 21). The EU
screen therefore never returns ``presumed_harmful``: a combined share of 50% or more is
reported as ``possible_dominance`` (paragraph 17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .units import FloatArray, Ownership, Shares

Status = Literal[
    "presumed_harmful",
    "warrants_scrutiny",
    "unlikely_concerns",
    "no_presumption",
    "possible_dominance",
]
RULESETS = ("us2023", "us2010", "eu2004")


@dataclass(frozen=True)
class Concentration:
    """HHI statistics of a merger, in points on the 0 to 10,000 scale."""

    hhi_pre: float
    hhi_post: float
    delta_hhi: float
    merged_share: float
    n_firms_pre: int
    n_firms_post: int


@dataclass(frozen=True)
class GuidelineResult:
    """Outcome of one guideline rule set."""

    ruleset: str
    name: str
    status: Status
    summary: str
    citation: str
    reference: str = ""


def hhi(shares: ArrayLike, ownership: Ownership | None = None) -> float:
    """Herfindahl-Hirschman index ``10^4 sum_f s_f^2`` of share fractions.

    With an ownership matrix the products of one owner are aggregated into a firm.
    """
    s = np.asarray(shares, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("shares must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(s)):
        raise ValueError("shares must be finite")
    if np.any(s < 0) or s.sum() > 1 + 1e-9:
        raise ValueError("shares must be a vector of non-negative fractions summing to at most 1")
    if ownership is None:
        return float(1e4 * np.sum(s**2))
    if not ownership.is_binary:
        raise ValueError("HHI needs binary ownership; aggregate partial ownership first")
    return float(1e4 * s @ (ownership.matrix @ s))


def delta_hhi_pair(share_a: float, share_b: float) -> float:
    """``2 s_a s_b 10^4``: the HHI increase when two firms with these shares merge."""
    if not (0 <= share_a <= 1 and 0 <= share_b <= 1 and share_a + share_b <= 1 + 1e-12):
        raise ValueError("shares must be fractions whose sum does not exceed 1")
    return 2e4 * share_a * share_b


def concentration(
    shares: Shares,
    pre: Ownership,
    post: Ownership,
    outside: Literal["atomistic", "exclude"] = "atomistic",
) -> Concentration:
    """HHI before and after the change from ``pre`` to ``post`` ownership.

    ``outside="atomistic"`` computes shares of the total market and treats the outside
    good as atomistic (it adds nothing to the HHI, the usual convention when the market
    includes a competitive fringe). ``outside="exclude"`` renormalises the inside shares
    to one. For shares without an outside good the two coincide.
    """
    s: FloatArray = shares.values if outside == "atomistic" else shares.within_values()
    hhi_pre = hhi(s, pre)
    hhi_post = hhi(s, post)
    changed = pre.merging_products(post)
    merged = float(s[changed].sum()) if changed.any() else 0.0
    n_pre = round(float(np.sum(1.0 / pre.matrix.sum(axis=1))))
    n_post = round(float(np.sum(1.0 / post.matrix.sum(axis=1))))
    return Concentration(hhi_pre, hhi_post, hhi_post - hhi_pre, merged, n_pre, n_post)


def screen_us2023(c: Concentration) -> GuidelineResult:
    """2023 U.S. Merger Guidelines, Guideline 1 (section 2.1).

    Presumption if post-merger HHI exceeds 1,800 and the increase exceeds 100, or if the
    merged firm's share exceeds 30% and the increase exceeds 100. Not meeting a threshold
    is not a safe harbour.
    """
    cite = "2023 Merger Guidelines, Guideline 1 (section 2.1, pp. 5-6)"
    ref = "Guideline 1"
    by_hhi = c.hhi_post > 1800 and c.delta_hhi > 100
    by_share = c.merged_share > 0.30 and c.delta_hhi > 100
    if by_hhi or by_share:
        reasons = []
        if by_hhi:
            reasons.append(
                f"post-merger HHI {c.hhi_post:,.0f} exceeds 1,800 and the increase "
                f"{c.delta_hhi:,.0f} exceeds 100"
            )
        if by_share:
            reasons.append(
                f"the merged firm's share {c.merged_share:.1%} exceeds 30% with an HHI "
                f"increase of {c.delta_hhi:,.0f}"
            )
        return GuidelineResult(
            "us2023",
            "U.S. Merger Guidelines 2023",
            "presumed_harmful",
            "Structural presumption triggered: " + "; ".join(reasons) + ".",
            cite,
            ref,
        )
    return GuidelineResult(
        "us2023",
        "U.S. Merger Guidelines 2023",
        "no_presumption",
        "Neither structural threshold is exceeded (post-merger HHI "
        f"{c.hhi_post:,.0f}, increase {c.delta_hhi:,.0f}, merged share "
        f"{c.merged_share:.1%}); this is not a safe harbour.",
        cite,
        ref,
    )


def screen_us2010(c: Concentration) -> GuidelineResult:
    """2010 U.S. Horizontal Merger Guidelines, section 5.3.

    Unconcentrated below 1,500, moderately concentrated 1,500 to 2,500, highly
    concentrated above 2,500; an HHI increase below 100 ordinarily needs no further
    analysis. Boundary values follow the wording: "below", "above", "more than".
    """
    cite = "2010 Horizontal Merger Guidelines, section 5.3 (pp. 18-19)"
    ref = "section 5.3"
    name = "U.S. Horizontal Merger Guidelines 2010"
    if c.delta_hhi < 100:
        return GuidelineResult(
            "us2010",
            name,
            "unlikely_concerns",
            f"Small change in concentration (increase {c.delta_hhi:,.0f} < 100): unlikely "
            "to have adverse competitive effects, ordinarily no further analysis.",
            cite,
            ref,
        )
    if c.hhi_post < 1500:
        return GuidelineResult(
            "us2010",
            name,
            "unlikely_concerns",
            f"Post-merger HHI {c.hhi_post:,.0f} is below 1,500 (unconcentrated): unlikely "
            "to have adverse competitive effects.",
            cite,
            ref,
        )
    if c.hhi_post <= 2500:
        if c.delta_hhi > 100:
            return GuidelineResult(
                "us2010",
                name,
                "warrants_scrutiny",
                f"Moderately concentrated (post-merger HHI {c.hhi_post:,.0f}) with an increase "
                f"of {c.delta_hhi:,.0f} > 100: potentially raises significant competitive "
                "concerns and often warrants scrutiny.",
                cite,
                ref,
            )
        return GuidelineResult(
            "us2010",
            name,
            "unlikely_concerns",
            f"Moderately concentrated (post-merger HHI {c.hhi_post:,.0f}) with an increase of "
            f"exactly {c.delta_hhi:,.0f}: neither below nor more than 100, so the guidelines' "
            "concern threshold for this band (more than 100) is not crossed.",
            cite,
            ref,
        )
    if c.delta_hhi > 200:
        return GuidelineResult(
            "us2010",
            name,
            "presumed_harmful",
            f"Highly concentrated (post-merger HHI {c.hhi_post:,.0f}) with an increase of "
            f"{c.delta_hhi:,.0f} > 200: presumed likely to enhance market power (rebuttable).",
            cite,
            ref,
        )
    return GuidelineResult(
        "us2010",
        name,
        "warrants_scrutiny",
        f"Highly concentrated (post-merger HHI {c.hhi_post:,.0f}) with an increase of "
        f"{c.delta_hhi:,.0f} between 100 and 200: potentially raises significant "
        "competitive concerns and often warrants scrutiny.",
        cite,
        ref,
    )


def screen_eu2004(c: Concentration) -> GuidelineResult:
    """EU Horizontal Merger Guidelines (2004), paragraphs 17 to 21.

    Indicators that concerns are unlikely: combined share not above 25% (paragraph 18),
    post-merger HHI below 1,000 (paragraph 19), HHI 1,000 to 2,000 with a delta below 250,
    or HHI above 2,000 with a delta below 150 (paragraph 20, subject to special
    circumstances that no number can capture). A combined share of 50% or more may in
    itself indicate dominance (paragraph 17) and is reported as ``possible_dominance``.
    Paragraph 21: none of these creates a presumption either way, so no status here is a
    presumption of harm.
    """
    cite = "EU Horizontal Merger Guidelines 2004, paragraphs 17-21"
    ref = "paras 17-21"
    name = "EU Horizontal Merger Guidelines 2004"
    if c.merged_share >= 0.50:
        return GuidelineResult(
            "eu2004",
            name,
            "possible_dominance",
            f"Combined share {c.merged_share:.1%} is 50% or more: may in itself be evidence "
            "of a dominant position (paragraph 17); this is not a presumption of harm.",
            cite,
            ref,
        )
    safe = []
    if 0 < c.merged_share <= 0.25:
        safe.append(f"combined share {c.merged_share:.1%} does not exceed 25% (paragraph 18)")
    if c.hhi_post < 1000:
        safe.append(f"post-merger HHI {c.hhi_post:,.0f} is below 1,000 (paragraph 19)")
    elif c.hhi_post <= 2000 and c.delta_hhi < 250:
        safe.append(
            f"post-merger HHI {c.hhi_post:,.0f} is between 1,000 and 2,000 with a delta of "
            f"{c.delta_hhi:,.0f} < 250 (paragraph 20)"
        )
    elif c.hhi_post > 2000 and c.delta_hhi < 150:
        safe.append(
            f"post-merger HHI {c.hhi_post:,.0f} is above 2,000 with a delta of "
            f"{c.delta_hhi:,.0f} < 150 (paragraph 20)"
        )
    if safe:
        return GuidelineResult(
            "eu2004",
            name,
            "unlikely_concerns",
            "Initial indicator that horizontal concerns are unlikely: " + "; ".join(safe) + ".",
            cite,
            ref,
        )
    extra = (
        " Share of 40% to 50% has led to dominance findings (paragraph 17)."
        if (0.40 <= c.merged_share < 0.50)
        else ""
    )
    return GuidelineResult(
        "eu2004",
        name,
        "warrants_scrutiny",
        f"No HHI or share indicator of absence of concerns applies (post-merger HHI "
        f"{c.hhi_post:,.0f}, delta {c.delta_hhi:,.0f}, combined share {c.merged_share:.1%}); "
        "further analysis needed." + extra,
        cite,
        ref,
    )


def screen(c: Concentration, rulesets: tuple[str, ...] = RULESETS) -> list[GuidelineResult]:
    """Apply the requested rule sets (``us2023``, ``us2010``, ``eu2004``)."""
    fns = {"us2023": screen_us2023, "us2010": screen_us2010, "eu2004": screen_eu2004}
    out = []
    for r in rulesets:
        if r not in fns:
            raise ValueError(f"unknown rule set {r!r}; choose from {RULESETS}")
        out.append(fns[r](c))
    return out
