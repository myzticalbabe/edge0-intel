"""Unit tests for ports_lock (C-P1-5). No servers needed."""
import os
import sys

import pytest

OWN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "scripts")
sys.path.insert(0, OWN)

import ports_lock
from ports_lock import acquire, cohabit, release, status


@pytest.fixture(autouse=True)
def isolate_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ports_lock, "LOCK_DIR", str(tmp_path))
    yield


def test_acquire_release_cycle():
    rec = acquire(8089, "unit")
    assert rec["owner"] == "unit"
    assert rec["pid"] == os.getpid()
    assert status()[8089]["owner"] == "unit"
    assert release(8089, "unit") is True
    assert status()[8089] is None


def test_double_acquire_denied_same_process():
    acquire(8089, "a")
    with pytest.raises(RuntimeError):
        acquire(8089, "b")
    release(8089, "a")


def test_release_wrong_owner_refused():
    acquire(8089, "a")
    assert release(8089, "b") is False
    assert status()[8089]["owner"] == "a"
    assert release(8089, "a") is True


def test_release_unheld_false():
    assert release(8091) is False


def _dead_pid() -> int:
    """PID that is certainly not alive on this OS.

    A literal pid=1 is init on Linux (and alive), so it cannot represent a
    stale holder there; reap a real child instead so the fixture is
    portable.
    """
    import subprocess
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait(timeout=60)
    return pid


def test_stale_lock_reaped():
    path = os.path.join(ports_lock.LOCK_DIR, "8090.json")
    os.makedirs(ports_lock.LOCK_DIR, exist_ok=True)
    import json
    pid = _dead_pid()
    assert not ports_lock._pid_alive(pid)
    json.dump({"port": 8090, "pid": pid, "owner": "dead",
               "since": "2000-01-01T00:00:00"}, open(path, "w"))
    rec = acquire(8090, "new")
    assert rec["owner"] == "new"


def test_cohabit_counts_siblings():
    acquire(8089, "a")
    acquire(8091, "b")
    assert cohabit(except_port=8089) == 1
    assert cohabit() == 2
    release(8089, "a")
    release(8091, "b")
