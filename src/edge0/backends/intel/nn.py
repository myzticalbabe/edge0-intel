"""Intel backend: nn module factory (Module, Linear, RMSNorm, silu, gelu)."""
from __future__ import annotations

import numpy as _np

from . import core as _core


def silu(x):
    a = _core._np.asarray(x, dtype=_core._np.float32) if hasattr(_core, "_np") else _np.asarray(x, dtype=_np.float32)
    return (a / (1.0 + _np.exp(-a))).astype(_np.float32)


def gelu(x):
    # exact erf gelu (matches prerouter head math)
    import math
    a = _np.asarray(x, dtype=_np.float32)
    return (0.5 * a * (1.0 + _core.erf(a / math.sqrt(2.0)))).astype(_np.float32)


class Module:
    def __call__(self, *args, **kwargs):
        return self.__call_impl__(*args, **kwargs) if hasattr(self, "__call_impl__") else self.forward(*args, **kwargs)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=_np.float16):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        rng = _np.random.default_rng(0)
        scale = 1.0 / max(1, in_features) ** 0.5
        self.weight = (rng.standard_normal((out_features, in_features)).astype(_np.float32) * scale).astype(dtype)
        self.bias = _np.zeros((out_features,), dtype=dtype) if bias else None

    def forward(self, x):
        y = _core.matmul(_np.asarray(x, dtype=_np.float32), self.weight.astype(_np.float32).T)
        if self.bias is not None:
            y = y + self.bias.astype(_np.float32)
        return y.astype(self.weight.dtype, copy=False)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = _np.ones((dim,), dtype=_np.float32)
        self.eps = eps

    def forward(self, x):
        a = _np.asarray(x, dtype=_np.float32)
        v = (a * a).mean(axis=-1, keepdims=True)
        return (a / _np.sqrt(v + self.eps) * self.weight).astype(x.dtype if hasattr(x, "dtype") else _np.float32)
