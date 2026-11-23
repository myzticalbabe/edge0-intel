"""Intel device selection: XPU (Arc GPU via Level Zero) with CPU fallback.

Works on:
  - Windows native (Arc 140T via Intel driver, torch.xpu if installed)
  - WSL2 Ubuntu (Level Zero / OpenCL via /dev/dri, oneAPI)

Env knobs:
  EDGE0_INTEL_DEVICE=cpu|xpu|auto  (default auto)
  ONEAPI_DEVICE_SELECTOR=level_zero:gpu  (passed through to SYCL runtime)
  EDGE0_BACKEND=intel  (selects this backend in edge0.backends)
"""
from __future__ import annotations

import os
import platform


def _have_torch_xpu() -> bool:
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def _have_dri() -> bool:
    # WSL2 / Linux GPU passthrough node
    try:
        import glob
        return len(glob.glob("/dev/dri/renderD*")) > 0
    except Exception:
        return False


def _have_opencl_intel() -> bool:
    # clinfo present + Intel platform string (checked by caller script too)
    try:
        import subprocess
        r = subprocess.run(["clinfo", "-l"], capture_output=True, text=True, timeout=5)
        return "Intel" in (r.stdout + r.stderr)
    except Exception:
        return False


def detect() -> dict:
    """Return capability dict; never raises."""
    info = {
        "os": platform.system(),
        "torch_xpu": _have_torch_xpu(),
        "dri": _have_dri(),
        "opencl_intel": _have_opencl_intel(),
        "oneapi_selector": os.environ.get("ONEAPI_DEVICE_SELECTOR", ""),
    }
    # GPU usable if torch.xpu works OR (Linux/DRI + Intel OpenCL present).
    # torch.xpu is the compute path; DRI/OpenCL alone means driver is there
    # but PyTorch Intel build is still needed for matmuls.
    info["gpu_driver_present"] = bool(info["dri"] or info["opencl_intel"])
    return info


def select_device() -> str:
    """'xpu' when GPU compute is usable, else 'cpu'."""
    want = os.environ.get("EDGE0_INTEL_DEVICE", "auto").lower()
    if want in ("cpu",):
        return "cpu"
    if want in ("xpu", "gpu"):
        return "xpu"
    # auto
    if _have_torch_xpu():
        return "xpu"
    return "cpu"


def describe() -> str:
    d = detect()
    return (
        f"os={d['os']} device={select_device()} "
        f"torch_xpu={d['torch_xpu']} dri={d['dri']} "
        f"opencl_intel={d['opencl_intel']} "
        f"selector={d['oneapi_selector'] or '(unset)'}"
    )
