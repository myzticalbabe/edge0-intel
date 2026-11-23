"""Q02 tests: agreement math, disagreement gate, blinding, drift chart.

All offline and deterministic. The seeded divergent pair (disagreement
0.70) is the Q02 accept criterion: the default 0.60 gate MUST fire on it.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import judge_track as jt  # noqa: E402
from judge_calibrate import self_test_pair  # noqa: E402


def test_pair_agreement_exact():
    assert jt.pair_agreement([True, True, False], [True, False, False]) \
        == pytest.approx(2 / 3)


def test_pair_agreement_rejects():
    with pytest.raises(ValueError):
        jt.pair_agreement([], [])
    with pytest.raises(ValueError):
        jt.pair_agreement([True], [True, False])
    with pytest.raises(ValueError):
        jt.pair_agreement([1, 0], [True, False])


def test_pairwise_matrix():
    m = jt.pairwise({"A": [True, False], "B": [True, True],
                     "C": [False, False]})
    assert m["A~B"] == 0.5
    assert m["A~C"] == 0.5
    assert m["B~C"] == 0.0
    with pytest.raises(ValueError):
        jt.pairwise({"only": [True]})


def test_seeded_divergent_pair_fires_default_gate():
    """Q02 accept criterion, part 1: seeded divergence fires the gate."""
    labels = self_test_pair(n=40)
    rate, pair = jt.max_disagreement(labels)
    assert rate == pytest.approx(0.70)
    assert pair == "J1~J2"
    gate = jt.gate_check(labels)  # default threshold 0.60
    assert gate["fired"] is True
    assert gate["rate"] == pytest.approx(0.70)


def test_identical_judges_pass():
    v = [True, False, True, True, False]
    gate = jt.gate_check({"J1": list(v), "J2": list(v)})
    assert gate["fired"] is False
    assert gate["rate"] == 0.0


def test_gate_threshold_validated():
    with pytest.raises(ValueError):
        jt.gate_check({"J1": [True], "J2": [False]}, threshold=1.5)


def test_blind_labels_deterministic_and_complete():
    labels, mapping = jt.blind({"judge-b": [True, False],
                                "judge-a": [True, True]})
    assert sorted(labels) == ["J1", "J2"]
    assert mapping == {"J1": "judge-a", "J2": "judge-b"}
    assert labels["J1"] == [True, True]
    # Agreement is label-agnostic: same votes, same value either way.
    assert list(jt.pairwise(labels).values()) == list(jt.pairwise(
        {"X": [True, True], "Y": [True, False]}).values())


def test_run_record_carries_no_identities():
    labels = self_test_pair(n=10)
    row = jt.run_record("r1", labels,
                        identities_hint="judge_label_map.json")
    blob = json.dumps(row)
    assert "http" not in blob and "model" not in blob.lower()
    assert row["label_map_file"] == "judge_label_map.json"
    assert row["kind"] == "calibration"
    with pytest.raises(ValueError):
        jt.run_record("r1", labels, identities_hint={"J1": "x"})


def test_drift_chart_renders_two_runs():
    r1 = jt.run_record("aaa", self_test_pair(n=40))
    twin = {k: list(v) for k, v in self_test_pair(n=40).items()}
    twin["J2"][0] = not twin["J2"][0]
    r2 = jt.run_record("bbb", twin)
    chart = jt.drift_chart(r1, r2)
    assert "aaa" in chart and "bbb" in chart
    assert "J1~J2" in chart
    assert "+0.0250" in chart  # one flipped vote of 40 moves 0.025


def test_drift_chart_marks_new_and_dropped():
    r1 = {"run_id": "a", "agreement": {"J1~J2": 0.5, "J1~J3": 0.4}}
    r2 = {"run_id": "b", "agreement": {"J1~J2": 0.6, "J2~J3": 0.9}}
    chart = jt.drift_chart(r1, r2)
    assert "NEW" in chart and "DROPPED" in chart
