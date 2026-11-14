"""GGUF and Safetensors surgery operators."""

def gguf_abliterate(input_path: str, output_path: str, strength: float = 1.0) -> dict:
    """Abliterate GGUF model with byte-exact metadata copy."""
    return {"status": "stub", "input": input_path, "output": output_path, "strength": strength}

def safetensors_abliterate(input_path: str, output_path: str) -> dict:
    """Abliterate safetensors model."""
    return {"status": "stub", "input": input_path, "output": output_path}