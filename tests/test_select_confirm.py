"""Q04 tests: deterministic disjoint split, overfit stopped at CONFIRM.

The Q04 accept criterion: a known-overfit config passes SELECT but is
stopped at CONFIRM. test_overfit_stopped_at_confirm is that criterion.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import select_confirm as sc  # noqa: E402


def test_partition_deterministic_disjoint_covering():
    items = list(range(100))
    s1, c1 = sc.partition(items, 0.70, 7)
    s2, c2 = sc.partition(items, 0.70, 7)
    assert s1 == s2 and c1 == c2
    assert len(s1) == 70 and len(c1) == 30
    assert sorted(s1 + c1) == items  # covering, no overlap by construction
    assert sc.split_id(0.70, 7) == "sc70-7"


def test_partition_rejects():
    with pytest.raises(ValueError):
        sc.partition([1], 0.7, 7)
    with pytest.raises(ValueError):
        sc.partition(list(range(10)), 1.0, 7)
    s, c = sc.partition(list(range(20)), 0.70, 7)
    assert (len(s), len(c)) == (14, 6)  # caps_20 scale


def test_overfit_stopped_at_confirm():
    """Q04 accept: tuned-to-SELECT passes SELECT, fails CONFIRM -> HOLD."""
    v = sc.confirm_check(sel_score=0.95, con_score=0.55,
                         sel_bar=0.80, con_bar=0.80,
                         select_hash="aaa", confirm_hash="bbb",
                         looks_used=3, max_looks=12)
    assert v["decision"] == "HOLD"
    assert v["reasons"] == ["confirm-below-bar"]


def test_genuine_promotes():
    v = sc.confirm_check(0.9, 0.85, 0.8, 0.8, "aaa", "bbb", 3, 12)
    assert v["decision"] == "PROMOTE"


def test_missing_hash_stops_not_seen():
    v = sc.confirm_check(0.95, 0.95, 0.8, 0.8, None, "b", 1, 12)
    assert v["decision"] == "STOP"
    assert v["reasons"] == ["missing-inputs-hash"]


def test_confirm_on_seen_data_stops():
    v = sc.confirm_check(0.95, 0.95, 0.8, 0.8, "same", "same", 1, 12)
    assert v["decision"] == "STOP"
    assert v["reasons"] == ["confirm-seen"]


def test_looks_budget_stops():
    v = sc.confirm_check(0.95, 0.95, 0.8, 0.8, "a", "b", 13, 12)
    assert v["decision"] == "STOP"
    assert v["reasons"] == ["looks-exhausted"]
    v = sc.confirm_check(0.5, 0.9, 0.8, 0.8, "a", "b", 12, 12)
    assert v["decision"] == "HOLD"  # budget exactly spent still decides


def test_cli_confirm_end_to_end(tmp_path, monkeypatch):
    log = tmp_path / "ledger.jsonl"
    pre = {"tag": "PREREG-c1", "kind": "prereg", "campaign": "c1",
           "falsifier": "f", "eval_N": 12, "promotion": "p"}
    sel = {"tag": "s1", "campaign": "c1", "phase": "select",
           "split_id": "sc70-7", "inputs_hash": "aaa", "f_caps": 0.95}
    con = {"tag": "t1", "campaign": "c1", "phase": "confirm",
           "split_id": "sc70-7", "inputs_hash": "bbb", "f_caps": 0.55}
    monkeypatch.setenv("EDGE0_LEDGER", str(log))
    with open(str(log), "w", encoding="utf-8") as f:
        for r in (pre, sel, con):
            f.write(json.dumps(r) + "\n")
    base = ["confirm", "--ledger", str(log), "--campaign", "c1",
            "--select-tag", "s1", "--confirm-tag", "t1",
            "--metric", "f_caps", "--sel-bar", "0.8",
            "--con-bar", "0.8"]
    assert sc.main(base) == 1  # HOLD: overfit stopped at CONFIRM
    con["inputs_hash"] = "aaa"
    with open(str(log), "w", encoding="utf-8") as f:
        for r in (pre, sel, con):
            f.write(json.dumps(r) + "\n")
    assert sc.main(base) == 2  # STOP: confirm ran on seen data
    con["inputs_hash"] = "bbb"
    con["f_caps"] = 0.9
    with open(str(log), "w", encoding="utf-8") as f:
        for r in (pre, sel, con):
            f.write(json.dumps(r) + "\n")
    assert sc.main(base) == 0  # PROMOTE: genuine clears both
