"""Model registry and refusal direction extraction."""

class ModelRegistry:
    """Registry of model artifacts (GGUF, safetensors, torch)."""
    def __init__(self):
        self.models = {}
    
    def register(self, name: str, path: str, format: str):
        self.models[name] = {"path": path, "format": format}

class RefusalDirection:
    """Refusal direction extraction and analysis."""
    def __init__(self, model_name: str, norm: float, dim: int):
        self.model_name = model_name
        self.norm = norm
        self.dim = dim
    
    def dose_response(self, strengths: list) -> dict:
        return {s: {"refusal_rate": 0.0} for s in strengths}