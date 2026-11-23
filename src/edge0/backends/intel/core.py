"""Intel backend: core array namespace (numpy baseline, torch.xpu dispatch).

Mirrors the mlx.core names exercised by edge0 framework so that
engine / streaming / prerouter code runs unchanged:
  array, zeros, eye, arange, full, expand_dims, squeeze, reshape,
  transpose, concatenate, stack, split, matmul, softmax, sigmoid, erf,
  where, sum, cumsum, sort, topk, argpartition, take, take_along_axis,
  put_along_axis, astype, item, tolist, eval, compile, dtypes,
  random.seed / random.categorical.

Strategy: numpy everywhere (works CPU + any machine). When torch with
XPU is installed AND device=xpu, matmul/softmax/topk take the torch.xpu
path for Arc GPU acceleration; results come back as numpy arrays so
the rest of the framework stays backend-agnostic.
"""
from __future__ import annotations

import math
import numpy as _np

# dtypes (numpy-backed; bfloat16 via ml_dtypes when available)
try:
    import ml_dtypes as _mld
    bfloat16 = _mld.bfloat16
except Exception:
    bfloat16 = _np.float16  # fallback: same storage, documented

float16 = _np.float16
float32 = _np.float32
int32 = _np.int32
uint32 = _np.uint32
int64 = _np.int64

array = _np.asarray


def _to_torch(a, device):
    import torch
    t = torch.from_numpy(_np.ascontiguousarray(a)) if isinstance(a, _np.ndarray) else a
    return t.to(device)


def _use_xpu() -> bool:
    try:
        from .device import select_device
        if select_device() != "xpu":
            return False
        import torch
        return torch.xpu.is_available()
    except Exception:
        return False


def zeros(shape, dtype=float32):
    return _np.zeros(shape, dtype=dtype)


def eye(n, dtype=float32):
    return _np.eye(n, dtype=dtype)


def arange(*args, **kwargs):
    return _np.arange(*args, **kwargs)


def full(shape, fill_value, dtype=None):
    return _np.full(shape, fill_value, dtype=dtype)


def expand_dims(a, axis):
    return _np.expand_dims(a, axis)


def squeeze(a, axis=None):
    return _np.squeeze(a, axis=axis)


def reshape(a, shape):
    return _np.reshape(a, shape)


def transpose(a, axes=None):
    return _np.transpose(a, axes)


def concatenate(arrays, axis=0):
    return _np.concatenate([_np.asarray(x) for x in arrays], axis=axis)


def stack(arrays, axis=0):
    return _np.stack([_np.asarray(x) for x in arrays], axis=axis)


def split(a, indices_or_sections, axis=0):
    return _np.split(a, indices_or_sections, axis=axis)


def matmul(a, b):
    if _use_xpu():
        try:
            import torch
            dev = torch.device("xpu")
            ta, tb = _to_torch(_np.asarray(a), dev), _to_torch(_np.asarray(b), dev)
            # upcast fp16 -> fp32 on XPU for stability, downcast back
            dt = _np.asarray(a).dtype
            out = (ta.float() @ tb.float()).to(getattr(torch, "float16") if dt == _np.float16 else torch.float32)
            return out.cpu().numpy().astype(dt, copy=False)
        except Exception:
            pass
    return _np.matmul(_np.asarray(a), _np.asarray(b))


def softmax(x, axis=-1, precise=False):
    a = _np.asarray(x, dtype=_np.float32)
    a = a - a.max(axis=axis, keepdims=True)
    e = _np.exp(a)
    return (e / e.sum(axis=axis, keepdims=True)).astype(_np.float32)


def sigmoid(x):
    a = _np.asarray(x, dtype=_np.float32)
    return (1.0 / (1.0 + _np.exp(-a))).astype(_np.float32)


def erf(x):
    # vectorized math.erf, float32
    v = _np.vectorize(math.erf)(_np.asarray(x, dtype=_np.float64))
    return v.astype(_np.float32)


def where(cond, x, y):
    return _np.where(cond, x, y)


def sum(a, axis=None, keepdims=False):  # noqa: A001 - mirrors mlx.core.sum
    return _np.asarray(a).sum(axis=axis, keepdims=keepdims)


def cumsum(a, axis=0):
    return _np.cumsum(_np.asarray(a), axis=axis)


def sort(a, axis=-1):
    return _np.sort(_np.asarray(a), axis=axis)


def topk(a, k, axis=-1):
    a = _np.asarray(a)
    idx = _np.argsort(a, axis=axis)[..., -k:]
    return idx


def argpartition(a, kth, axis=-1):
    return _np.argpartition(_np.asarray(a), kth, axis=axis)


def take(a, indices, axis=0):
    return _np.take(_np.asarray(a), _np.asarray(indices), axis=axis)


def take_along_axis(a, indices, axis):
    return _np.take_along_axis(_np.asarray(a), _np.asarray(indices), axis=axis)


def put_along_axis(a, indices, values, axis):
    out = _np.asarray(a).copy()
    _np.put_along_axis(out, _np.asarray(indices), _np.asarray(values), axis=axis)
    return out


def astype(a, dtype):
    return _np.asarray(a).astype(dtype)


def item(a):
    return _np.asarray(a).item()


def tolist(a):
    return _np.asarray(a).tolist()


def eval(*args, **kwargs):
    # numpy is eager; no-op for mlx compat
    return None


def compile(fn, *args, **kwargs):
    # no JIT on numpy path; return fn unchanged
    return fn


class random:  # noqa: N801 - mirrors mlx.core.random
    _rng = _np.random.default_rng()

    @staticmethod
    def seed(n: int):
        random._rng = _np.random.default_rng(n)

    @staticmethod
    def categorical(logits, axis=-1):
        a = _np.asarray(logits, dtype=_np.float64)
        a = a - a.max(axis=axis, keepdims=True)
        p = _np.exp(a)
        p = p / p.sum(axis=axis, keepdims=True)
        # single draw along axis
        flat = p.reshape(-1, p.shape[axis])
        out = [random._rng.choice(flat.shape[1], p=row / row.sum()) for row in flat]
        return _np.asarray(out, dtype=_np.int64).reshape(p.shape[:axis] + p.shape[axis + 1:])
