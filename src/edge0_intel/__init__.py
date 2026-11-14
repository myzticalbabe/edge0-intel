"""Edge0-Intel — Tensor Shard Evolution & GGUF Surgery."""

__version__ = "3.16.0"
__author__ = "myzticalbabe"
__license__ = "SOUL FORGE v1.0"

from .surgery import gguf_abliterate, safetensors_abliterate
from .evolution import EvolutionEngine, SVIAdmission
from .models import ModelRegistry, RefusalDirection

__all__ = [
    "gguf_abliterate",
    "safetensors_abliterate",
    "EvolutionEngine",
    "SVIAdmission",
    "ModelRegistry",
    "RefusalDirection",
]