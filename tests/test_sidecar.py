"""Unit tests for the shared provenance-sidecar writer (C-P0-3).

Fast, no model, no server. Run: python -m pytest tests/test_sidecar.py -q
"""
import json
import os
import sys

import pytest

OWN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "scripts")
sys.path.insert(0, OWN)

from write_sidecar import SCHEMA, sha256_file, write_build_sidecar


@pytest.fixture()
def workdir(tmp_path):
    return tmp_path


def _blob(workdir, name="x.bin", data=b"hello"):
    p = workdir / name
    p.write_bytes(data)
    return str(p)


def test_sha256_known_vector(workdir):
    p = _blob(workdir)
    assert sha256_file(p) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_missing_returns_none(workdir):
    assert sha256_file(str(workdir / "nope.bin")) is None


def test_write_fills_bytes_and_hash(workdir):
    dst = _blob(workdir)
    side = write_build_sidecar(dst, {"src": "unit", "strength": 1.0})
    assert side == dst + ".build.json"
    rec = json.loads(open(side, encoding="utf-8").read())
    assert rec["schema"] == SCHEMA
    assert rec["out_bytes"] == 5
    assert rec["out_sha256"] == sha256_file(dst)
    assert rec["src"] == "unit"
    assert "time" in rec


def test_write_hashes_refusal_dir(workdir):
    dst = _blob(workdir, "model.gguf", b"weights")
    rdir = _blob(workdir, "refusal_dir.npy", b"dir")
    rec = json.loads(open(
        write_build_sidecar(dst, {"refusal_dir": rdir}), encoding="utf-8").read())
    assert rec["refusal_sha256"] == sha256_file(rdir)


def test_write_missing_artifact_records_none(workdir):
    dst = str(workdir / "ghost.gguf")
    rec = json.loads(open(
        write_build_sidecar(dst, {"backfilled": True}), encoding="utf-8").read())
    assert rec["out_bytes"] is None
    assert rec["out_sha256"] is None
    assert rec["backfilled"] is True


def test_caller_supplied_hashes_not_overwritten(workdir):
    dst = _blob(workdir)
    rec = json.loads(open(write_build_sidecar(dst, {
        "out_bytes": 999, "out_sha256": "keepme",
    }), encoding="utf-8").read())
    assert rec["out_bytes"] == 999
    assert rec["out_sha256"] == "keepme"
