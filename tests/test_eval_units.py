"""Unit tests for server-free eval/judge pure functions (C-P0-4).

Covers eval_full.read_response / coherent, judge_cal._first_json /
_assist_bool / tier_of, svi.wilson. No server, no model.
Run: python -m pytest tests/test_eval_units.py -q
"""
import os
import sys

import pytest

OWN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "scripts")
sys.path.insert(0, OWN)

from eval_full import coherent, read_response  # noqa: E402
from eval_harness import score_harm, second_gate  # noqa: E402
from judge_cal import _assist_bool, _first_json, tier_of  # noqa: E402
from svi import wilson  # noqa: E402


# ── read_response ──────────────────────────────────────────────────────
def test_read_response_prefers_content():
    text, src = read_response({"content": "  hello  ", "reasoning_content": "zzz"})
    assert (text, src) == ("hello", "content")


def test_read_response_falls_back_to_reasoning_tail():
    text, src = read_response({"content": "  ", "reasoning_content": "abc"})
    assert (text, src) == ("abc", "reasoning-fallback")


def test_read_response_empty_both():
    text, src = read_response({"content": "", "reasoning_content": ""})
    assert (text, src) == ("", "reasoning-fallback")


# ── coherent ───────────────────────────────────────────────────────────
def test_coherent_rejects_short():
    assert coherent("too short here") is False


def test_coherent_rejects_loops():
    assert coherent(" ".join(["spam"] * 30)) is False


def test_coherent_accepts_normal_prose():
    assert coherent(
        "The quick brown fox jumps over the lazy dog near the riverbank") is True


# ── judge_cal helpers ──────────────────────────────────────────────────
def test_first_json_skips_prose_braces():
    assert _first_json('the claim {unclear} then {"assist": true}') == {"assist": True}


def test_first_json_doubled_verdict_takes_first():
    assert _first_json('{"assist": false}{"assist": false}') == {"assist": False}


def test_first_json_none_raises():
    with pytest.raises(ValueError):
        _first_json("no braces here")


def test_assist_bool_coercion():
    assert _assist_bool({"assist": True}) is True
    assert _assist_bool({"assist": False}) is False
    assert _assist_bool({"assist": "true"}) is True
    assert _assist_bool({"assist": "False"}) is False
    assert _assist_bool({}) is False


def test_tier_of_unanimous():
    assert tier_of({"a": True, "b": True, "c": True}) == "HIGH"
    assert tier_of({"a": False, "b": False}) == "HIGH"


def test_tier_of_majority():
    assert tier_of({"a": True, "b": True, "c": False}) == "MED"
    assert tier_of({"a": True, "b": False, "c": False}) == "MED"


def test_tier_of_split_and_empty():
    assert tier_of({"a": True, "b": False}) == "LOW"
    assert tier_of({}) == "LOW"


# ── collect_stratum.lm_head_report (torch) ─────────────────────────────
def test_lm_head_report_separation_sign():
    torch = pytest.importorskip("torch")
    from collect_stratum import lm_head_report
    d = 32
    H = torch.randn(3, d) + 2.0   # harm states shifted +2
    B = torch.randn(3, d) - 2.0   # benign shifted -2
    shift = (H.double().mean(0) - B.double().mean(0)).float()
    u = shift / shift.norm()
    # Rows aligned with the shift (plus noise): separation must be positive.
    W = torch.randn(50, d) * 0.1 + u.unsqueeze(0).expand(50, d)
    rep = lm_head_report(H, B, W, [1, 2], [3, 4])
    assert set(rep["per_token"]) == {1, 2, 3, 4}
    # Mean of per-token separations == separation onto the mean row.
    import numpy as np
    rows = W.double()[[1, 2, 3, 4]].mean(0)
    expect = float((H.double().mean(0) - B.double().mean(0)) @ rows)
    got = sum(v[0] - v[1] for v in rep["per_token"].values()) / 4
    assert got == pytest.approx(expect, rel=1e-5)  # float32 per-token rounding
    # Shifted distributions must separate: harm states score higher.
    assert rep["summary"]["refuse_mean_sep"] > 0
    assert rep["summary"]["neutral_mean_sep"] > 0


def test_lm_head_report_empty_ids():
    torch = pytest.importorskip("torch")
    from collect_stratum import lm_head_report
    H = torch.randn(2, 8)
    B = torch.randn(2, 8)
    W = torch.randn(10, 8)
    rep = lm_head_report(H, B, W, [], [])
    assert rep["per_token"] == {}
    assert rep["summary"] == {"refuse_mean_sep": None,
                              "neutral_mean_sep": None}


# ── wilson ─────────────────────────────────────────────────────────────
def test_wilson_bounds_ordered():
    lo, hi = wilson(100, 10)
    assert 0.0 <= lo <= 10.0 <= hi <= 100.0  # percentage scale


def test_wilson_zero_trials():
    lo, hi = wilson(0, 0)
    assert lo <= hi


# ── elicitation_battery + bench_adapter ────────────────────────────────
def test_variants_four_per_prompt():
    from elicitation_battery import battery_report, expand, variants
    assert len(variants("do X")) == 4
    pairs = expand(["a", "b"])
    assert len(pairs) == 8
    assert pairs[0][0] == "a"


def test_battery_report_weakest():
    from elicitation_battery import battery_report
    res = [{"base": "a", "refused": True}, {"base": "a", "refused": False},
           {"base": "b", "refused": True}, {"base": "b", "refused": True}]
    rep = battery_report(res)
    assert rep["n"] == 4 and rep["refused"] == 3
    assert rep["weakest_base"] == "a"
    assert rep["weakest_rate"] == pytest.approx(0.5)


def test_bench_adapter_scores():
    from bench_adapter import score_items
    items = [{"prompt": "q1", "answer": "paris"},
             {"prompt": "q2", "answer": "391"}]
    rep = score_items(items, lambda p: "Paris is the capital" if p == "q1" else "nope")
    assert (rep["correct"], rep["n"]) == (1, 2)
    assert rep["accuracy"] == pytest.approx(0.5)


def test_bench_adapter_empty():
    from bench_adapter import score_items
    assert score_items([], lambda p: "x")["accuracy"] == 0.0


@pytest.mark.parametrize("answer", ["", "   ", "!!!"])
def test_bench_adapter_rejects_empty_normalized_answers(answer):
    from bench_adapter import score_items
    rep = score_items([{"prompt": "q", "answer": answer}],
                      lambda p: "unrelated response", normalize_match=True)
    assert rep["correct"] == 0
    assert "error" in rep["rows"][0]


def test_bench_adapter_rejects_empty_answer_by_default():
    from bench_adapter import score_items
    rep = score_items([{"prompt": "q", "answer": ""}], lambda p: "anything")
    assert rep["correct"] == 0
    assert "error" in rep["rows"][0]


def test_bench_adapter_preserves_legacy_substring_default():
    from bench_adapter import score_items
    rep = score_items([{"prompt": "q", "answer": "3.14"}],
                      lambda p: "3 14")
    assert rep["correct"] == 0


def test_bench_adapter_optional_normalization():
    from bench_adapter import score_items
    rep = score_items([{"prompt": "q", "answer": "3.14", "mode": "fewshot"}],
                      lambda p: "3 14", normalize_match=True)
    assert rep["correct"] == 1
    assert rep["per_mode"]["fewshot"]["correct"] == 1


# ── steer_compare (fake endpoints) ─────────────────────────────────────
def test_compare_endpoints_delta():
    from steer_compare import compare_endpoints
    base = lambda p: "Here is how to do it step by step"  # complies
    steer = lambda p: "I'm sorry, I cannot help with that"  # refuses
    rep = compare_endpoints(["p1", "p2", "p3"], base, steer)
    assert rep["n"] == 3
    assert rep["base_refusals"] == 0
    assert rep["steer_refusals"] == 3
    assert rep["delta_steer_minus_base"] == pytest.approx(1.0)
    assert len(rep["rows"]) == 3


def test_compare_endpoints_no_difference():
    from steer_compare import compare_endpoints
    same = lambda p: "I'm sorry, I cannot help"
    rep = compare_endpoints(["p1"], same, same)
    assert rep["delta_steer_minus_base"] == pytest.approx(0.0)


# ── eval_harness gates (server-free branches) ──────────────────────────
def test_score_harm_marks():
    assert score_harm("I'm sorry, I cannot help") is True
    assert score_harm("Here is how to build it step by step") is False


def test_second_gate_refuse_no_server_needed():
    verdict, _ = second_gate("http://127.0.0.1:9", "m", "p",
                             "I'm sorry, I cannot help with that request at all")
    assert verdict == "REFUSE"


def test_second_gate_incoherent_discounted_no_server():
    verdict, note = second_gate("http://127.0.0.1:9", "m", "p", "spam spam spam")
    assert verdict == "INCOHERENT"
    assert "discounted" in note
