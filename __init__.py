"""
TileMix package initialization.

Model integration:
    from TileMix.models.tilemix_llama import inject_tilemix_into_llama
    model = inject_tilemix_into_llama(model)

    from TileMix.models.tilemix_qwen import inject_tilemix_into_qwen
    model = inject_tilemix_into_qwen(model)
"""

__all__ = [
    "inject_tilemix_into_llama",
    "inject_tilemix_into_qwen",
]

try:
    from .models.tilemix_llama import inject_tilemix_into_llama
    from .models.tilemix_qwen import inject_tilemix_into_qwen
except Exception:
    pass

