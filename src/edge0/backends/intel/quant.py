"""Intel backend: quantized gather kernels.

Reference (bit-compatible math, not bit-identical assembly):
  gather_qmm(x, w_int4, scales, biases, rhs_indices, ...) ->
    dequant selected expert rows -> matmul.

Matches mlx gather_qmm semantics: w [num_experts, out, in] int-packed,
rows gathered by rhs_indices, then dequant (affine, group_size, bits).
Runs on CPU (numpy) and Arc GPU (torch.xpu matmul in core).
"""
from __future__ import annotations

import numpy as _np

from . import core as _core


def _dequant_affine(w, scales, biases, group_size=64, bits=4):
    w = _np.asarray(w)
    scales = _np.asarray(scales, dtype=_np.float32)
    biases = _np.asarray(biases, dtype=_np.float32)
    # Generic path: w already unpacked to int values [E, out, in].
    # Packed uint32 path (32/bits vals per word) unpacked on the fly.
    if w.dtype in (_np.uint32, _np.int32) and w.ndim == 3:
        # assume last dim packed
        vals_per = 32 // bits
        mask = (1 << bits) - 1
        E, O, packed_in = w.shape
        inn = packed_in * vals_per
        out = _np.empty((E, O, inn), dtype=_np.float32)
        for g in range(packed_in):
            word = w[:, :, g].astype(_np.uint32)
            for v in range(vals_per):
                out[:, :, g * vals_per + v] = ((word >> (v * bits)) & mask).astype(_np.float32)
        w = out
    else:
        w = w.astype(_np.float32)
    # group affine: scale/bias per [E, out, in//group]
    if scales.ndim == 3 and w.shape[-1] % scales.shape[-1] == 0:
        rep = w.shape[-1] // scales.shape[-1]
        scales = _np.repeat(scales, rep, axis=-1)
        biases = _np.repeat(biases, rep, axis=-1)
    return w * scales + biases


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
               group_size=64, bits=4, mode="affine", sorted_indices=False):
    x = _np.asarray(x, dtype=_np.float32)
    idx = _np.asarray(rhs_indices).ravel()
    dq = _dequant_affine(w, scales, biases, group_size, bits).astype(_np.float32)
    # dq [E, out, in]; select experts
    sel = dq[idx]  # [K, out, in]
    if transpose:
        # x [..., in] @ [K, in, out]
        sel = _np.transpose(sel, (0, 2, 1))
        outs = [_core.matmul(x, sel[k]) for k in range(sel.shape[0])]
    else:
        outs = [_core.matmul(x, sel[k].T) for k in range(sel.shape[0])]
    return _np.stack(outs, axis=0)


def swiglu(up, gate):
    up = _np.asarray(up, dtype=_np.float32)
    gate = _np.asarray(gate, dtype=_np.float32)
    return (gate / (1.0 + _np.exp(-gate)) * up).astype(_np.float32)
