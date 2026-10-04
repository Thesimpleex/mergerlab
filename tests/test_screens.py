import numpy as np
import pytest

from mergerlab.screens import (
    Concentration,
    concentration,
    delta_hhi_pair,
    hhi,
    screen,
    screen_eu2004,
    screen_us2010,
    screen_us2023,
)
from mergerlab.units import Ownership, Shares


def conc(pre, delta, share=0.2):
    return Concentration(pre, pre + delta, delta, share, 5, 4)


def test_hhi_examples_from_the_guidelines():
    assert hhi([0.2] * 5) == pytest.approx(2000)  # 2023 Guidelines, footnote 13
    assert hhi([1 / 6] * 6) == pytest.approx(1666.667, abs=1e-3)
    assert hhi([0.4, 0.2, 0.15, 0.15, 0.10]) == pytest.approx(2550)  # EU guidelines, footnote 18
    assert delta_hhi_pair(0.30, 0.15) == pytest.approx(900)  # EU guidelines, footnote 19


def test_delta_hhi_equals_difference_of_hhi():
    s = np.array([0.30, 0.15, 0.25, 0.20, 0.10])
    pre = Ownership.from_owners(list("ABCDE"))
    post = pre.merged(["A", "B"])
    c = concentration(Shares.within(s, "revenue"), pre, post)
    assert c.hhi_pre == pytest.approx(hhi(s))
    assert c.delta_hhi == pytest.approx(delta_hhi_pair(0.30, 0.15))
    assert c.merged_share == pytest.approx(0.45)
    assert (c.n_firms_pre, c.n_firms_post) == (5, 4)


def test_outside_good_conventions():
    s = Shares.total([0.3, 0.2, 0.1], "revenue")
    pre = Ownership.from_owners(["A", "B", "C"])
    post = pre.merged(["A", "B"])
    atom = concentration(s, pre, post, outside="atomistic")
    excl = concentration(s, pre, post, outside="exclude")
    assert atom.hhi_pre == pytest.approx(1e4 * (0.09 + 0.04 + 0.01))
    assert excl.hhi_pre == pytest.approx(1e4 * (0.25 + 0.111111 + 0.027778), rel=1e-4)
    assert excl.hhi_pre > atom.hhi_pre


def test_us2023_thresholds():
    assert screen_us2023(conc(1700, 150)).status == "presumed_harmful"  # post 1850 > 1800
    assert screen_us2023(conc(1700, 100)).status == "no_presumption"  # delta must exceed 100
    assert screen_us2023(conc(1700, 101)).status == "presumed_harmful"
    assert screen_us2023(conc(1650, 150)).status == "no_presumption"  # post exactly 1800
    assert screen_us2023(conc(900, 150, share=0.31)).status == "presumed_harmful"
    assert screen_us2023(conc(900, 150, share=0.30)).status == "no_presumption"
    assert screen_us2023(conc(900, 100, share=0.40)).status == "no_presumption"


def test_us2010_categories():
    assert screen_us2010(conc(1800, 99)).status == "unlikely_concerns"
    assert screen_us2010(conc(1000, 300)).status == "unlikely_concerns"  # post 1300 < 1500
    assert screen_us2010(conc(1800, 150)).status == "warrants_scrutiny"  # moderately concentrated
    assert screen_us2010(conc(2000, 100)).status == "unlikely_concerns"  # moderate needs > 100
    assert screen_us2010(conc(2500, 150)).status == "warrants_scrutiny"  # highly, 100-200
    assert screen_us2010(conc(2500, 201)).status == "presumed_harmful"
    assert screen_us2010(conc(2500, 200)).status == "warrants_scrutiny"


def test_eu2004_indicators():
    assert screen_eu2004(conc(500, 400, share=0.30)).status == "unlikely_concerns"  # post < 1000
    assert screen_eu2004(conc(1500, 200, share=0.35)).status == "unlikely_concerns"  # delta < 250
    assert screen_eu2004(conc(1500, 300, share=0.35)).status == "warrants_scrutiny"
    assert screen_eu2004(conc(2500, 149, share=0.35)).status == "unlikely_concerns"
    assert screen_eu2004(conc(2500, 150, share=0.35)).status == "warrants_scrutiny"
    assert screen_eu2004(conc(3000, 800, share=0.20)).status == "unlikely_concerns"  # share <= 25%
    assert screen_eu2004(conc(3000, 800, share=0.55)).status == "possible_dominance"
    assert "40% to 50%" in screen_eu2004(conc(3000, 800, share=0.45)).summary


@pytest.mark.parametrize(
    ("post", "expected"),
    [(1499, "unlikely_concerns"), (1500, "warrants_scrutiny"), (1501, "warrants_scrutiny")],
)
def test_us2010_unconcentrated_boundary_is_1500(post, expected):
    assert screen_us2010(conc(post - 150, 150)).status == expected


@pytest.mark.parametrize(
    ("post", "expected"),
    [(2500, "warrants_scrutiny"), (2501, "warrants_scrutiny")],
)
def test_us2010_moderately_concentrated_summary_follows_the_status(post, expected):
    res = screen_us2010(conc(post - 150, 150))
    assert res.status == expected and "warrants scrutiny" in res.summary
    exact = screen_us2010(conc(1900, 100))
    assert exact.status == "unlikely_concerns" and "warrants scrutiny" not in exact.summary


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (0.25, "unlikely_concerns"),
        (0.2501, "warrants_scrutiny"),
        (0.30, "warrants_scrutiny"),
        (0.4999, "warrants_scrutiny"),
        (0.50, "possible_dominance"),
    ],
)
def test_eu2004_share_boundaries(share, expected):
    assert screen_eu2004(conc(3000, 800, share=share)).status == expected


@pytest.mark.parametrize(
    ("post", "delta", "expected"),
    [
        (999, 300, "unlikely_concerns"),  # HHI below 1,000
        (1000, 300, "warrants_scrutiny"),
        (1749, 249, "unlikely_concerns"),  # 1,000-2,000 band, delta below 250
        (1750, 250, "warrants_scrutiny"),
        (1800, 251, "warrants_scrutiny"),
        (2000, 249, "unlikely_concerns"),  # upper end of the band
        (2000, 250, "warrants_scrutiny"),
        (2150, 149, "unlikely_concerns"),  # above 2,000, delta below 150
        (2150, 150, "warrants_scrutiny"),
        (2001, 200, "warrants_scrutiny"),
    ],
)
def test_eu2004_hhi_and_delta_boundaries(post, delta, expected):
    assert screen_eu2004(conc(post - delta, delta, share=0.35)).status == expected


def test_screen_runs_all_rulesets_and_rejects_unknown():
    res = screen(conc(1700, 150))
    assert [r.ruleset for r in res] == ["us2023", "us2010", "eu2004"]
    assert all(r.citation for r in res)
    with pytest.raises(ValueError, match="unknown rule set"):
        screen(conc(1700, 150), ("us1992",))


def test_hhi_rejects_nan_and_empty_input():
    with pytest.raises(ValueError, match="finite"):
        hhi([0.5, float("nan")])
    with pytest.raises(ValueError, match="non-empty"):
        hhi([])


def test_hhi_input_validation():
    with pytest.raises(ValueError, match="fractions"):
        hhi([0.7, 0.5])
    with pytest.raises(ValueError, match="fractions"):
        delta_hhi_pair(0.7, 0.5)
