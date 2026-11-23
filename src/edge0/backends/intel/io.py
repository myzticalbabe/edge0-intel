"""Intel backend: io (model/tokenizer/tensor-store loading, numpy mmap).

Same contract as backends/mlx/io.py but returns numpy arrays so it runs
on CPU and on Arc GPU (via torch.xpu in core) without mlx.
"""
from __future__ import annotations

import json
import mmap
import struct

import numpy as _np

from edge0.backends.base import TensorStore


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)


def open_shards(model_dir: str) -> list:
    import glob
    import os
    from edge0.streaming.mmap import SafetensorsMmap
    shards = []
    for path in sorted(glob.glob(os.path.join(os.fspath(model_dir), "model*.safetensors"))):
        shards.append(SafetensorsMmap(path))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors shards under {model_dir}")
    return shards


def load_model(model_path, lazy=True, strict=False, model_config=None, get_model_classes=None):
    # Intel path defers skeleton build to transformers; MoE streaming works
    # at the shard/mmap level (open_shards) so no resident full-model load.
    raise NotImplementedError(
        "intel load_model: use open_shards() + streaming layers; "
        "full resident load stays on llama.cpp SYCL server for now")


def load_safetensors(path: str, dtype=None) -> dict:
    store = SafeTensorsStore(path)
    try:
        out = {}
        for name in store.keys():
            arr = store.get(name)
            if dtype is not None and arr.dtype != dtype:
                arr = arr.astype(dtype)
            out[name] = arr
        return out
    finally:
        store.close()


class SafeTensorsStore(TensorStore):
    """mmap-backed safetensors store returning numpy arrays (zero copy)."""

    def __init__(self, path: str):
        self._path = path
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        self._metadata = header.pop("__metadata__", {})
        self._entries = {
            name: {
                "offset": meta["data_offsets"][0] + 8 + header_len,
                "size": meta["data_offsets"][1] - meta["data_offsets"][0],
                "dtype": meta["dtype"],
                "shape": tuple(meta["shape"]),
            }
            for name, meta in header.items()
        }
        self._keys = sorted(self._entries)
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def path(self) -> str:
        return self._path

    def keys(self) -> list[str]:
        return list(self._keys)

    def metadata(self) -> dict:
        return dict(self._metadata)

    def get(self, name: str):
        e = self._entries.get(name)
        if e is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        buf = _np.frombuffer(self._mm, dtype=_np.uint8, count=e["size"], offset=e["offset"])
        arr = _np.frombuffer(buf, dtype=_DTYPES[e["dtype"]], count=int(_np.prod(e["shape"], dtype=int)) if e["shape"] else 1)
        return arr.reshape(e["shape"])

    def close(self):
        try:
            self._mm.close()
        finally:
            self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


try:
    import ml_dtypes as _mld
    _BF16 = _mld.bfloat16
except Exception:
    _BF16 = _np.float16

_DTYPES = {
    "F64": _np.float64, "F32": _np.float32, "F16": _np.float16,
    "BF16": _BF16, "I64": _np.int64, "I32": _np.int32,
    "I16": _np.int16, "I8": _np.int8, "U8": _np.uint8, "BOOL": _np.bool_,
}


def open_tensor_store(path: str) -> TensorStore:
    if path.endswith(".safetensors"):
        return SafeTensorsStore(path)
    raise ValueError(f"unsupported adapter format: {path!r} (only .safetensors)")
