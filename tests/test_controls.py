"""Q03 tests: control verdicts on seeded readings, no servers needed.

The Q03 accept criterion: a sweep with a sabotaged control is flagged
UNVALIDATED. test_sabotaged_dead_control and test_broken_stock_path
are that criterion, asserted directly.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import controls as ct  # noqa: E402


def rd(stock, dead):
    s = {"refused": stock[0], "caps_ok": stock[1], "eval_N": stock[2]}
    d = {"refused": dead[0], "caps_ok": dead[1], "eval_N": dead[2]}
    return {"stock": s, "dead": d}


def test_clean_sweep_validated():
    v = ct.check_controls(rd((8, 18, 100), (10, 17, 20 * 5)))
    assert v["validated"] is True
    assert v["reasons"] == ["controls-nominal"]


def test_sabotaged_dead_control():
    """Dead shard swapped for a live one: behaves like a champion while
    stock behaves like stock. MUST be UNVALIDATED (Q03 accept)."""
    v = ct.check_controls(rd((10, 18, 100), (95, 20, 100)))
    assert v["validated"] is False
    assert "dead-diverged" in v["reasons"]
    assert v["detail"]["divergence"]["refused"] == pytest.approx(0.85)


def test_broken_stock_path():
    """Stock cannot do caps: the serve path is broken, not the shards."""
    v = ct.check_controls(rd((0, 4, 100), (1, 3, 100)))
    assert v["validated"] is False
    assert "stock-path-dead" in v["reasons"]


def test_missing_control_fails():
    v = ct.check_controls({"stock": {"refused": 1, "caps_ok": 18,
                                     "eval_N": 100}})
    assert v["validated"] is False
    assert "missing-dead" in v["reasons"]
    v = ct.check_controls({})
    assert v["validated"] is False
    assert "missing-stock" in v["reasons"] and "missing-dead" in v["reasons"]


def test_empty_and_mismatched_n():
    v = ct.check_controls(rd((0, 0, 0), (0, 0, 0)))
    assert v["validated"] is False
    assert "empty-stock" in v["reasons"]
    v = ct.check_controls(rd((8, 18, 100), (10, 17, 20)))
    assert v["validated"] is False
    assert "n-mismatch" in v["reasons"]


def test_tolerance_boundary():
    ok = ct.check_controls(rd((10, 18, 100), (25, 18, 100)), tol=0.15)
    assert ok["validated"] is True  # 0.15 exactly is within tol
    bad = ct.check_controls(rd((10, 18, 100), (26, 18, 100)), tol=0.15)
    assert bad["validated"] is False


def test_cli_check_and_close(tmp_path, monkeypatch):
    log = tmp_path / "ledger.jsonl"
    rows = [
        {"tag": "CTL-stock", "refused": 8, "caps_ok": 18, "eval_N": 100},
        {"tag": "CTL-dead", "refused": 10, "caps_ok": 17, "eval_N": 100},
        "not json{{{",
        {"tag": "CTL-dead", "refused": 95, "caps_ok": 20, "eval_N": 100},
    ]
    with open(str(log), "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    # Latest dead row is the sabotaged one -> exit 2.
    assert ct.main(["check", "--ledger", str(log), "--stock-tag",
                    "CTL-stock", "--dead-tag", "CTL-dead"]) == 2
    v = ct.check_controls(ct.readings_from_ledger(str(log), "CTL-stock",
                                                  "CTL-dead"))
    assert ct.main(["close", "--ledger", str(log), "--campaign", "t",
                    "--verdict-json", json.dumps(v), "--stock-tag",
                    "CTL-stock", "--dead-tag", "CTL-dead"]) == 0
    last = ct.latest_tag_row(str(log), "SWEEP-t")
    assert last["kind"] == "sweep" and last["validated"] is False
