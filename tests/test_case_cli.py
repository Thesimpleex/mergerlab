import re
from pathlib import Path

import pytest
from rich.console import Console

from mergerlab.case import CaseError, analyze, load_case, parse_case
from mergerlab.cli import main, render_analysis
from mergerlab.report import render_report
from mergerlab.summary import summarize

EXAMPLE = Path(__file__).parents[1] / "examples" / "heinz_beech_nut.toml"


def minimal(**over):
    data = {
        "market": {"share_basis": "revenue", "outside_share": 0.1},
        "products": [
            {"name": "A", "share": 0.5, "margin": 0.3},
            {"name": "B", "share": 0.4},
        ],
        "merger": {"parties": ["A", "B"]},
        "models": {"include": ["logit"]},
    }
    data.update(over)
    return data


def test_example_case_loads_and_analyses():
    case = load_case(EXAMPLE)
    assert [m.kind for m in case.models] == ["logit", "nested_logit", "ces", "linear", "pcaids"]
    an = analyze(case, draws=40)
    assert all(o.ok for o in an.simulation.outcomes)
    assert an.monte_carlo is not None and an.monte_carlo.draws == 40
    assert set(an.hypothetical_monopolist) <= {m.kind for m in case.models}
    s = summarize(an)
    assert s.point_range is not None and s.point_range[0] < s.point_range[1]


def test_parse_minimal_case():
    case = parse_case(minimal())
    assert case.market.shares.outside == pytest.approx(0.1)
    assert case.parties == ("A", "B")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.pop("merger"), "missing 'merger'"),
        (lambda d: d["merger"].update(parties=["A", "Z"]), "unknown owner"),
        (lambda d: d["market"].update(share_basis="volume"), "share_basis"),
        (lambda d: d["market"].update(outside_share=0.5), "sum to 1"),
        (lambda d: d["models"].update(include=["translog"]), "unknown demand model"),
        (
            lambda d: d["products"][0].update(margin_absolute=0.2),
            "either margin or margin_absolute",
        ),
        (lambda d: d["merger"].update(efficiency=1.5), "efficiency"),
        (lambda d: d.update(products=[]), "at least one"),
        (lambda d: d.update(screens={"rulesets": ["us1992"]}), "unknown rule set"),
        (lambda d: d.update(uncertainty={"sigma_range": [0.2]}), "two-element"),
    ],
)
def test_case_errors_are_explicit(mutate, message):
    data = minimal()
    mutate(data)
    with pytest.raises(CaseError, match=message):
        parse_case(data)


def test_invalid_toml_is_a_case_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml")
    with pytest.raises(CaseError, match="invalid TOML"):
        load_case(bad)


def test_cli_simulate_prints_summary_and_writes_report(tmp_path, capsys):
    out = tmp_path / "memo.html"
    code = main(["simulate", str(EXAMPLE), "--draws", "30", "--report", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert "Heinz / Beech-Nut" in text
    assert "Price effects by demand form" in text
    assert "Nested logit" in text and "PCAIDS" in text
    assert out.exists()


def test_cli_screen(capsys):
    code = main(["screen", "--shares", "30", "20", "15", "15", "10", "10", "--merge", "1", "2"])
    text = capsys.readouterr().out
    assert code == 0
    assert "U.S. Merger Guidelines 2023" in text and "presumption" in text


def test_cli_reports_errors_with_exit_code_2(tmp_path, capsys):
    code = main(["simulate", str(tmp_path / "missing.toml")])
    assert code == 2
    assert "mergerlab:" in capsys.readouterr().err


def test_report_is_self_contained_and_numbers_match_summary():
    an = analyze(load_case(EXAMPLE), draws=40)
    html = render_report(an)
    assert not re.search(r"(src|href)=[\"']https?://", html)
    assert "<script" not in html
    assert "<svg" in html
    s = summarize(an)
    for row in s.rows:
        assert f"{100 * row.average_change:+.1f}%" in html
    assert f"{an.simulation.concentration.hhi_post:,.0f}" in html
    assert "not legal, antitrust or investment advice" in html


def _write_case(tmp_path, text):
    path = tmp_path / "case.toml"
    path.write_text(text)
    return path


UNEQUAL_PRICES_CASE = """
[market]
share_basis = "revenue"
outside_share = 0.1
outside_price = 1.0

[[products]]
name = "A"
price = 2.0
share = 0.40
margin = 0.30

[[products]]
name = "B"
price = 1.5
share = 0.30

[[products]]
name = "C"
price = 0.8
share = 0.20

[merger]
parties = ["A", "B"]

[models]
include = ["logit", "ces", "linear", "pcaids"]

[models.linear]
fill_margins = "logit"

[models.pcaids]
own_elasticity = -2.5
market_elasticity = -1.3
"""


def test_cli_runs_a_case_with_unequal_prices_under_every_demand_form(tmp_path, capsys):
    path = _write_case(tmp_path, UNEQUAL_PRICES_CASE)
    code = main(["simulate", str(path), "--draws", "0"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "Traceback" not in out.err
    for label in ("Logit", "CES", "Linear", "PCAIDS"):
        assert label in out.out
    assert "metrics failed" not in out.out


FAILING_CASE = """
[market]
outside_share = 0.1

[[products]]
name = "A"
share = 0.5
margin = 0.3

[[products]]
name = "B"
share = 0.4

[merger]
parties = ["A", "B"]

[models]
include = ["logit", "linear"]
"""


def test_cli_prints_failure_reasons_and_exits_one_when_no_form_calibrates(tmp_path, capsys):
    path = _write_case(tmp_path, FAILING_CASE)
    memo = tmp_path / "memo.html"
    code = main(["simulate", str(path), "--draws", "0", "--report", str(memo)])
    out = capsys.readouterr()
    assert code == 1
    assert "outside_price" in out.out
    assert "no demand form could be calibrated" in out.err
    html = memo.read_text()
    assert "Across demand forms it is" not in html
    assert "<b>-</b>" not in html


def test_cli_table_is_not_truncated_when_piped(tmp_path, capsys):
    code = main(["simulate", str(EXAMPLE), "--draws", "20"])
    text = capsys.readouterr().out
    assert code == 0
    assert "\u2026" not in text
    assert "MC median [5%, 95%]" in text and "Consumer harm" in text and "CMCR" in text


def test_effects_table_drops_party_columns_before_headline_columns_on_narrow_terminals():
    an = analyze(load_case(EXAMPLE), draws=20)
    console = Console(width=80, force_terminal=True, color_system=None, highlight=False)
    with console.capture() as cap:
        render_analysis(console, an)
    text = cap.get()
    assert "\u2026" not in text
    assert "CMCR" in text and "Synergies" in text
    header = next(line for line in text.splitlines() if "Demand form" in line)
    assert "Gerber" not in header and "Average" in header


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--shares", "0.65", "0.174", "0.154", "--merge", "2", "3"], "did you mean 65 17.4 15.4"),
        (["--shares", "65", "17.4", "15.4", "--merge", "1", "9"], "between 1 and 3"),
        (["--shares", "100", "--merge", "1", "2"], "at least two firms"),
        (["--shares", "60", "30", "--merge", "1", "1"], "two different firms"),
        (["--shares", "70", "50", "--merge", "1", "2"], "at most 100"),
        (["--shares", "5", "3", "--merge", "1", "2"], "percentages"),
    ],
)
def test_cli_screen_rejects_inconsistent_input(argv, message, capsys):
    code = main(["screen", *argv])
    err = capsys.readouterr().err
    assert code == 2
    assert message in err
    assert "list index" not in err


def test_cli_screen_says_when_it_infers_an_outside_share(capsys):
    assert main(["screen", "--shares", "40", "30", "20", "--merge", "1", "2"]) == 0
    assert "remaining 10% is treated as an atomistic outside good" in capsys.readouterr().out
    assert main(["screen", "--shares", "50", "30", "20", "--merge", "1", "2"]) == 0
    assert "outside good" not in capsys.readouterr().out


def test_cli_rejects_negative_draws(capsys):
    assert main(["simulate", str(EXAMPLE), "--draws", "-5"]) == 2
    assert "--draws" in capsys.readouterr().err


def test_non_numeric_case_values_are_case_errors_naming_the_key():
    data = minimal()
    data["products"][0]["share"] = "big"
    with pytest.raises(CaseError, match=r"share of A.*number"):
        parse_case(data)
    data = minimal()
    data["uncertainty"] = {"draws": 2.5}
    with pytest.raises(CaseError, match=r"draws.*whole number"):
        parse_case(data)
    data = minimal()
    data["merger"]["efficiency"] = True
    with pytest.raises(CaseError, match="efficiency"):
        parse_case(data)


def test_empty_model_list_is_a_case_error():
    data = minimal()
    data["models"] = {"include": []}
    with pytest.raises(CaseError, match="at least one demand model"):
        parse_case(data)


def test_pcaids_anchor_is_reported_not_silently_defaulted():
    data = minimal()
    data["products"][0]["margin"] = 0.4
    data["models"] = {"include": ["pcaids"]}
    out = analyze(parse_case(data), draws=0).simulation.outcomes[0]
    assert out.ok and out.calibration is not None
    assert any(note.startswith("anchored on A") for note in out.calibration.notes)
