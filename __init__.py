__all__ = [
    "inject_tilemix_into_llama",
    "inject_tilemix_into_qwen",
]

try:
    from .models.tilemix_llama import inject_tilemix_into_llama
    from .models.tilemix_qwen import inject_tilemix_into_qwen
except Exception:
    pass
