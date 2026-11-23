"""Smoke test Intel backend: core/nn/quant/device + prerouter math (numpy path).

Converted to real pytest functions 2026-09-24 (FU8) — was module-level asserts
(pytest collected nothing). Env vars stay at module top: edge0 reads them at
import time, before the test body runs.
"""
import os
import sys

os.environ["EDGE0_BACKEND"] = "intel"
os.environ.setdefault("EDGE0_INTEL_DEVICE", "cpu")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_upstream", "src"))

from edge0.backends.intel import core, nn, quant  # noqa: E402
from edge0.backends.intel.device import describe, select_device  # noqa: E402


def test_matmul_and_softmax():
    a = core.matmul(np.eye(4, dtype=np.float32),
                    np.full((4, 4), 2.0, dtype=np.float32))
    assert float(a.sum()) == 32.0, a.sum()
    s = core.softmax(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert abs(float(s.sum()) - 1.0) < 1e-5


def test_prerouter_top2_select():
    logits = np.array([[0.1, 2.0, 0.5, 1.0]], dtype=np.float32)
    g = core.softmax(logits, precise=True)
    idx = core.argpartition(g, kth=-2)[..., -2:]
    assert idx.shape == (1, 2)


def test_linear_swiglu_shapes():
    lin = nn.Linear(8, 4, bias=False)
    y = lin(np.random.default_rng(0).standard_normal((2, 8)).astype(np.float16))
    assert y.shape == (2, 4)
    q = quant.swiglu(np.ones((2, 4), np.float32), np.ones((2, 4), np.float32))
    assert q.shape == (2, 4)


def test_device_describe():
    assert isinstance(select_device(), str)
    assert isinstance(describe(), str)
