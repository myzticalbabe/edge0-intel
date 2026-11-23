"""Q01 gate tests: pre-registration record/check/require.

Ledger isolation: EDGE0_LEDGER points at a tmp file per test, so the real
evolve_log.jsonl is never touched.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import prereg  # noqa: E402


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "test_ledger.jsonl"
    monkeypatch.setenv("EDGE0_LEDGER", str(p))
    monkeypatch.setenv("EDGE0_PREREG_STRICT", "1")
    return p


def test_record_then_check(ledger):
    row = prereg.record_prereg("ling-t1", "refusal<0.5 kills dose",
                               100, "refusal>=0.9 and caps>=0.8")
    assert row["tag"] == "PREREG-ling-t1"
    assert row["eval_N"] == 100
    assert prereg.find_prereg("ling-t1")["falsifier"] == "refusal<0.5 kills dose"
    assert prereg.require_prereg("ling-t1")["tag"] == "PREREG-ling-t1"
    lines = ledger.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1 and json.loads(lines[0])["kind"] == "prereg"


def test_check_missing(ledger):
    assert prereg.find_prereg("nope") is None
    with pytest.raises(prereg.PreregMissing):
        prereg.require_prereg("nope")


def test_fields_required(ledger):
    with pytest.raises(ValueError):
        prereg.record_prereg("", "f", 100, "p")
    with pytest.raises(ValueError):
        prereg.record_prereg("c", "  ", 100, "p")
    with pytest.raises(ValueError):
        prereg.record_prereg("c", "f", 100, "")
    with pytest.raises(ValueError):
        prereg.record_prereg("c", "f", 0, "p")
    with pytest.raises(ValueError):
        prereg.record_prereg("c", "f", "abc", "p")
    assert prereg.find_prereg("c") is None  # nothing half-written


def test_latest_prereg_wins(ledger):
    prereg.record_prereg("c", "first", 50, "p1")
    prereg.record_prereg("c", "second", 100, "p2")
    assert prereg.find_prereg("c")["falsifier"] == "second"


def test_malformed_lines_tolerated(ledger, monkeypatch):
    prereg.record_prereg("c", "f", 10, "p")
    with open(str(ledger), "a", encoding="utf-8") as f:
        f.write("not json{{{\n\n")
    assert prereg.find_prereg("c")["falsifier"] == "f"


def test_non_strict_backfill_warns(ledger, monkeypatch, capsys):
    monkeypatch.setenv("EDGE0_PREREG_STRICT", "0")
    assert prereg.require_prereg("ghost") is None
    assert "WARNING" in capsys.readouterr().out


def test_cli_record_check(ledger, capsys):
    assert prereg.main(["record", "--campaign", "cli1", "--falsifier", "f",
                        "--n", "20", "--promotion", "p"]) == 0
    assert prereg.main(["check", "--campaign", "cli1"]) == 0
    assert prereg.main(["check", "--campaign", "missing"]) == 1
