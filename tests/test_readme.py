"""The README quickstart runs, and its headline numbers match docs/results.json."""

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RESULTS = json.loads((ROOT / "docs" / "results.json").read_text())


def _blocks(lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", README, flags=re.S)


def test_quickstart_runs_and_prints_the_documented_output():
    code = _blocks("python")[0]
    expected = _blocks("text")[0]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "README quickstart", "exec"), {})
    assert buffer.getvalue().strip() == expected.strip()


def test_convention_trap_snippet_runs():
    code = _blocks("python")[1]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "README convention trap", "exec"), {})
    out = buffer.getvalue()
    assert "0.375" in out
    assert "[0.16666667 0.12698413]" in out


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _headline_strings() -> list[str]:
    forms = RESULTS["demand_forms"]["forms"]
    trap = RESULTS["convention_trap"]
    gate = RESULTS["solver_gate"]
    mil = RESULTS["miller_reproduction"]
    koh = RESULTS["koh_first_order"]["rows"]
    orc = RESULTS["oracles"]
    dem = RESULTS["demand_forms"]
    out = [
        f"{dem['hhi_pre']:,.0f} to {dem['hhi_post']:,.0f} (+{dem['delta_hhi']:,.0f})",
        f"merged share is {_pct(dem['merged_share'])}",
        f"logit {_pct(forms['logit']['price_change_average'])}",
        f"CES {_pct(forms['ces']['price_change_average'])}",
        f"PCAIDS {_pct(forms['pcaids']['price_change_average'])}",
        f"linear {_pct(forms['linear']['price_change_average'])}",
        f"nested logit {_pct(forms['nested_logit']['price_change_average'])}",
        f"median of {_pct(dem['ensemble_p05_p50_p95'][1])}",
        f"{_pct(dem['ensemble_p05_p50_p95'][0])} to {_pct(dem['ensemble_p05_p50_p95'][2])}",
        f"{_pct(dem['cmcr_range'][0])} (CES) to {_pct(dem['cmcr_range'][1])} (nested logit)",
        f"{dem['synergy_range_musd'][0]:.1f} to {dem['synergy_range_musd'][1]:.1f} million USD",
        f"from {dem['harm_range_musd'][0]:.1f} (PCAIDS) to {dem['harm_range_musd'][1]:.1f}",
        f"{_pct(forms['nested_logit']['monte_carlo_failure_share'])} of its Monte Carlo",
        f"{_pct(dem['failure_share_overall'])} of the 5,000",
        f"Gerber margin of {_pct(forms['nested_logit']['implied_margins']['Gerber'], 0)}",
        f"{_pct(trap['cmcr_wrong_a'])} for A instead of the correct {_pct(trap['cmcr_correct_a'])}",
        f"(B: {_pct(trap['cmcr_wrong_b'])} instead of {_pct(trap['cmcr_correct_b'])})",
        f"between +{100 * gate['trace']['naive_max_increase'][-1]:.1f}%",
        f"+{100 * gate['trace']['naive_max_increase'][-2]:.1f}%",
        f"the equilibrium is +{100 * gate['trace']['pyblp_max_increase']:.1f}%",
        f"fails the residual gate in {gate['random']['naive_fails_gate']} "
        f"({_pct(gate['random']['naive_fail_share'], 0)})",
        f"{mil['redrawn_not_rationalisable']} non-rationalisable",
        f"{mil['mape']['upp_vs_logit'] * 100:.2f} percentage points (paper 0.6)",
        f"{mil['mape']['upp_vs_linear'] * 100:.2f} (paper 2.2)",
        f"{mil['mape']['foa_vs_logit'] * 100:.2f} percentage points under logit",
        f"{orc['pyblp']['max_rel_error']:.1e}",
        f"{orc['antitrust']['epstein_rubinfeld_max_abs_error_pct_points']:.1e}",
        f"{orc['antitrust']['werden_table1_max_abs_error_pct_points']:.1e}",
        f"{orc['antitrust']['logit_price_change_max_abs_error_pct_points']:.1e}",
    ]
    m = mil["medians_ours"]
    out += [
        f"HHI {m['hhi_pre']:,.0f} / {m['hhi_post']:,.0f} / change {m['delta_hhi']:,.0f}",
        f"upward pricing pressure {m['upp']:.3f}",
        f"logit price effect {m['effect_logit']:.3f}",
        f"linear {m['effect_linear']:.3f}",
    ]
    out.append(
        "consumer harm "
        + ", ".join(f"{r['harm_first_order_musd']:.2f}" for r in koh[:-1])
        + f" and {koh[-1]['harm_first_order_musd']:.2f} million USD"
    )
    shares = [r["first_order_share_of_simulated"] for r in koh]
    out.append(f"the formula captures {100 * min(shares):.0f}% to {100 * max(shares):.0f}%")
    ratios = [r["first_order_share_of_exact_rivals_fixed"] for r in koh]
    out.append("(ratios " + ", ".join(f"{x:.2f}" for x in ratios[:-1]) + f" and {ratios[-1]:.2f})")
    rs = [r["rival_response_share_of_simulated"] for r in koh]
    out.append(f"accounts for {100 * min(rs):.0f}% to {100 * max(rs):.0f}% of the full harm")
    g = [r["gerber_price_change_simulated"] for r in koh]
    out.append(f"(+{100 * min(g):.1f}% to +{100 * max(g):.1f}%)")
    out.append(f"only {koh[0]['curvature_gap_musd']:.1f} million USD of the gap at sigma = 1.5")
    nl = forms["nested_logit"]
    q = nl["failure_share_by_nesting_parameter_quartile"]
    out.append(
        f"from {100 * q[0]:.0f}% to {100 * q[1]:.0f}%, {100 * q[2]:.0f}% and {100 * q[3]:.0f}%"
    )
    out.append(
        f"{nl['nesting_parameter_mean_successful']:.2f} among successful draws against "
        f"{nl['nesting_parameter_mean_failed']:.2f}"
    )
    out.append(f"nested logit: {nl['monte_carlo_successful_draws']}")
    rnd = gate["random"]
    out.append(
        f"fails in {rnd['damped_fails_gate']} of the {rnd['markets']} "
        f"({_pct(rnd['damped_fail_share'])})"
    )
    cm15 = trap["market_elasticity_minus_1_5"]
    out.append(f"the quantity diversion is {cm15['quantity_diversion_ab_true']:.3f}, not the")
    out.append(f"{cm15['quantity_diversion_ab_via_revenue_conversion']:.3f} that")
    out.append(
        f"the CMCR is {_pct(cm15['cmcr_true'][0])} and {_pct(cm15['cmcr_true'][1])} for A and B, "
        f"against {_pct(cm15['cmcr_via_revenue_conversion'][0])} and "
        f"{_pct(cm15['cmcr_via_revenue_conversion'][1])}"
    )
    out.append(
        f"median error is {100 * mil['mape']['foa_linear_passthrough_vs_logit_truth']:.2f} and "
        f"{100 * mil['mape']['foa_logit_passthrough_vs_linear_truth']:.2f} percentage points"
    )
    ant = orc["antitrust"]
    out.append(f"{ant['pcaids_market_elasticity_max_abs_error_pct_points']:.1e} percentage points")
    out.append(f"{ant['share_diversion_cmcr_max_abs_error_pct_points']:.1e} percentage points")
    out.append(
        "("
        + ", ".join(f"{r['harm_simulated_musd']:.1f}" for r in koh[:-1])
        + f" and {koh[-1]['harm_simulated_musd']:.1f} million USD)"
    )
    return out


@pytest.mark.parametrize("needle", _headline_strings())
def test_readme_number_matches_results(needle):
    assert needle in README
