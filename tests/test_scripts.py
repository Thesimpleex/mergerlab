"""The experiment scripts that feed docs/results.json (fast ones)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import exp_convention_trap
import exp_divestiture
import exp_koh
import exp_oracles


def test_oracle_agreement_is_at_numerical_precision():
    res = exp_oracles.run()
    assert res["pyblp"]["max_rel_error"] < 1e-12
    assert len(res["pyblp"]["scenarios"]) == 5
    r = res["antitrust"]
    assert r["epstein_rubinfeld_max_abs_error_pct_points"] < 1e-6
    assert r["werden_table1_max_abs_error_pct_points"] < 1e-9
    assert r["werden_table1_range_pct"] == pytest.approx([3.5088, 77.7778], abs=1e-4)
    assert r["cmcr_bertrand_doc_max_abs_error_pct_points"] < 1e-9
    assert r["logit_price_change_max_abs_error_pct_points"] < 1e-3


def test_convention_trap_numbers():
    res = exp_convention_trap.run()
    assert res["cmcr_correct_a"] == pytest.approx(0.16667, abs=5e-6)
    assert res["cmcr_wrong_a"] == pytest.approx(0.329, abs=5e-4)
    assert res["revenue_diversion_ab"] == pytest.approx(0.375)
    assert res["conversion_roundtrip_error"] < 1e-12
    assert "quantity diversion" in res["typed_api_error_message"]


def test_koh_experiment_reproduces_table_and_separates_the_rival_response():
    res = exp_koh.run()
    for row in res["rows"]:
        assert row["abs_diff_to_published_musd"] < 0.005
        assert 0.4 < row["first_order_share_of_simulated"] < 0.7
        assert 0.93 < row["first_order_share_of_exact_rivals_fixed"] < 1.02
        assert row["rival_response_share_of_simulated"] > 0.3
        total = (
            row["harm_first_order_musd"] + row["curvature_gap_musd"] + row["rival_response_musd"]
        )
        assert total == pytest.approx(row["harm_simulated_musd"])


def test_divestiture_experiment_ranks_by_remaining_harm():
    res = exp_divestiture.run()
    for model in res["models"].values():
        harms = [o["remaining_harm_musd"] for o in model["options"]]
        assert harms == sorted(harms)
        assert all(0 <= h < model["baseline_harm_musd"] for h in harms)
