"""Q05 tests: file hashes, env fingerprint, replay, backfill audit,
backfilled-cannot-promote (Q05 accept, part 2). No servers needed."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import provenance as pv  # noqa: E402
import select_confirm as sc  # noqa: E402
import eval_full as ef  # noqa: E402


def test_file_sha_full_and_prefix(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(os.urandom(3 << 20))
    raw = open(str(p), "rb").read()
    assert pv.file_sha(str(p)) == hashlib.sha256(raw).hexdigest()[:16]
    assert pv.file_sha(str(p), limit=1 << 20) == \
        hashlib.sha256(raw[:1 << 20]).hexdigest()[:16]


def test_env_fingerprint_stable():
    a = pv.env_fingerprint()
    b = pv.env_fingerprint()
    assert a == b
    assert len(a["env_fp"]) == 16 and a["python"]


def test_replay_full_phase_ok(tmp_path):
    want, note = pv.expected_inputs_hash({"phase": "full"})
    assert note == "file-bytes-only" and len(want) == 16
    row = {"tag": "t", "phase": "full", "split_id": "", "inputs_hash": want,
           "genome": {}}
    v = pv.replay_check(row)
    assert v["inputs"] == "ok" and v["outputs"] == "unverifiable"


def test_replay_select_phase_exact():
    from select_confirm import partition, split_id, items_hash
    harm = pv._eval_items("harmful_100.json")[:100]
    caps = pv._eval_items("caps_20.json")  # eval never slices caps
    hs, _ = partition(harm, 0.70, 7)
    cs, _ = partition(caps, 0.70, 7)
    h = items_hash(hs + cs)
    row = {"tag": "t", "phase": "select", "split_id": split_id(0.70, 7),
           "split_frac": 0.70, "split_seed": 7, "slice_n": 100,
           "eval_N": 70, "caps_n": 14, "inputs_hash": h, "genome": {}}
    assert pv.replay_check(row)["inputs"] == "ok"
    row["inputs_hash"] = "deadbeefdeadbeef"
    assert pv.replay_check(row)["inputs"] == "mismatch"


def test_backfilled_cannot_promote():
    """Q05 accept, part 2: backfilled lineage STOPs even when fit clears."""
    v = sc.confirm_check(sel_score=0.99, con_score=0.99,
                         sel_bar=0.8, con_bar=0.8,
                         select_hash="a", confirm_hash="b",
                         looks_used=1, max_looks=12, never_promote=True)
    assert v["decision"] == "STOP"
    assert v["reasons"] == ["never-promote"]
    v = sc.confirm_check(0.99, 0.99, 0.8, 0.8, "a", "b", 1, 12)
    assert v["decision"] == "PROMOTE"  # clean lineage unaffected


def test_audit_flags_backfilled_sidecars():
    found = pv.audit_backfilled()
    bf = [f for f in found if f["kind"] == "backfilled"]
    # Ground truth: exactly one reconstructed sidecar. The 17 layer
    # sidecars (edited=3, the 3 abliterated tensors) are clean.
    assert len(bf) == 1
    assert bf[0]["sidecar"] == "ling-abl-s05.gguf.build.json"
    layer = [f for f in found
             if f["sidecar"].startswith("ling-layer-")]
    assert layer == []


def test_build_link_full_sha_and_backfill(tmp_path):
    gguf = tmp_path / "shard.gguf"
    gguf.write_bytes(os.urandom(3 << 20))
    side = {"backfilled": True, "dst": "x", "edited": 1}
    with open(str(gguf) + ".build.json", "w", encoding="utf-8") as f:
        json.dump(side, f)
    info = ef.build_link(str(gguf))
    raw = open(str(gguf), "rb").read()
    assert info["build_full"] == hashlib.sha256(raw).hexdigest()[:16]
    assert info["build_sha"] == hashlib.sha256(raw[:1 << 20]).hexdigest()[:16]
    assert info["recipe"]["backfilled"] is True


def test_cli_replay_exits(tmp_path):
    log = tmp_path / "ledger.jsonl"
    want, _ = pv.expected_inputs_hash({"phase": "full"})
    rows = [{"tag": "good", "phase": "full", "split_id": "",
             "inputs_hash": want, "genome": {}},
            {"tag": "bad", "phase": "full", "split_id": "",
             "inputs_hash": "0" * 16, "genome": {}}]
    with open(str(log), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert pv.main(["replay", "--ledger", str(log), "--tag", "good"]) == 0
    assert pv.main(["replay", "--ledger", str(log), "--tag", "bad"]) == 1
    assert pv.main(["replay", "--ledger", str(log), "--tag", "ghost"]) == 2
