"""Intel backend assembly: exposes core/nn/io/quant + BackendImpl.

Usage:
  EDGE0_BACKEND=intel python -m edge0 ...
  EDGE0_INTEL_DEVICE=auto|xpu|cpu
"""
from __future__ import annotations

from edge0.backends.intel import core, io, nn, quant  # noqa: F401
from edge0.backends.intel.device import describe, select_device


class BackendImpl:
    name = "intel"
    description = "Intel Arc / CPU via numpy + torch.xpu (Level Zero/SYCL)"

    @property
    def version(self) -> str:
        try:
            import torch
            v = torch.__version__
        except Exception:
            v = "numpy-only"
        return f"{v} device={select_device()} [{describe()}]"
