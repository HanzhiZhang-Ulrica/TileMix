import torch
import os
from torch import Tensor
from typing import Optional
from attention.triton.tilemix import tilemix_attention_with_quant
from attention.triton.tilemix_bitmask_optimized import (
    tilemix_attention_bitmask_optimized_with_quant,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name) or default).strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def tilemix_varlen_attention(
    q: Tensor, k: Tensor, v: Tensor,
    block_m: int, block_n: int,
    *,
    block_n_mask: Optional[int] = None,
    causal: bool,
    scale: float,
    precision_map_all: Tensor,
    candidates: list[tuple[int, int]],
    flat_lengths: list[int],
    bias: Optional[Tensor] = None,
) -> Tensor:
    B, Hq, Tq, D = q.shape
    _, Hk, Tk, Dk = k.shape
    _, Hv, Tv, Dv = v.shape
    assert Dk == D
    assert Dv == D
    assert Hk == Hv, "K and V must have the same number of heads (GQA support)"

    Q = q.permute(0, 2, 1, 3).reshape(B * Tq, Hq, D).contiguous()
    K = k.permute(0, 2, 1, 3).reshape(B * Tk, Hk, D).contiguous()
    V = v.permute(0, 2, 1, 3).reshape(B * Tk, Hv, D).contiguous()

    device = Q.device
    cu_q = torch.arange(0, (B + 1) * Tq, Tq, device=device,
                        dtype=torch.int32).contiguous()
    cu_k = torch.arange(0, (B + 1) * Tk, Tk, device=device,
                        dtype=torch.int32).contiguous()

    pm = precision_map_all
    assert pm.dtype == torch.uint64, f"precision_map dtype must be uint64, got {pm.dtype}"
    if block_n_mask is None:
        block_n_mask = block_n
    if block_n_mask % block_n != 0:
        raise ValueError(f"block_n_mask ({block_n_mask}) must be a multiple of block_n ({block_n})")
    
    if Tq != Tk:
        raise ValueError(
            f"Tq ({Tq}) != Tk ({Tk}). Bitmask was created assuming they're equal. "
            f"This will cause incorrect indexing in the kernel."
        )
    
    m_blocks_total = (Tq + block_m - 1) // block_m
    n_mask_blocks_total = (Tk + block_n_mask - 1) // block_n_mask
    if n_mask_blocks_total > 64:
        raise ValueError(
            f"64-bit precision mask requires n_mask_blocks<=64, got n_mask_blocks={n_mask_blocks_total} "
            f"(Tk={Tk}, block_n_mask={block_n_mask})"
        )
    expected_pm_elems = Hk * m_blocks_total
    assert pm.numel() == expected_pm_elems, (
        f"precision_map length mismatch: got {pm.numel()}, expected {expected_pm_elems} "
        f"(Hk={Hk} KV heads × m_blocks={m_blocks_total}, n_mask_blocks={n_mask_blocks_total})"
    )

    if bias is not None:
        bias = bias.to(device=device, dtype=q.dtype).contiguous()
        if bias.shape != (B, Hq, Tq, Tk):
            raise ValueError(
                f"Bias shape mismatch: got {bias.shape}, expected {(B, Hq, Tq, Tk)}"
            )
    
    fuse_quant = _env_flag("TILEMIX_FUSE_QUANT")
    use_bitmask_optimized = _env_flag("TILEMIX_BITMASK_OPTIMIZED")
    score_only_pv = _env_flag("TILEMIX_SCORE_ONLY_PV", "1")
    if use_bitmask_optimized and fuse_quant:
        raise NotImplementedError(
            "TILEMIX_BITMASK_OPTIMIZED currently supports pre-quantized "
            "operands only; disable TILEMIX_FUSE_QUANT for a matched result"
        )

    if use_bitmask_optimized:
        # Select mixed/explicit routing policy for bitmask-optimized path.
        routing_mode = (
            os.getenv("TILEMIX_BITMASK_ROUTING_MODE") or "mixed"
        ).strip().lower()
        Out, _ = tilemix_attention_bitmask_optimized_with_quant(
            Q, K, V,
            cu_q, cu_k,
            max_seqlens_q=Tq, max_seqlens_k=Tk,
            precision_bitmask=pm,
            block_m=block_m, block_n=block_n,
            block_n_mask=block_n_mask,
            # Match the existing launcher's quantization granularity.
            BLKQ=128, BLKK=64,
            causal=causal, sm_scale=scale,
            bias=bias,
            routing_mode=routing_mode,
            run_length=False,
            scalar_scales=(block_m == 128 and block_n == 64),
            int8_cache_ca=False,
            int8_use_fp16_v=score_only_pv,
        )
    else:
        Out = tilemix_attention_with_quant(
            Q, K, V,
            cu_q, cu_k,
            max_seqlens_q=Tq, max_seqlens_k=Tk,
            precision_bitmask=pm,
            block_m=block_m, block_n=block_n,
            block_n_mask=block_n_mask,
            causal=causal, sm_scale=scale,
            bias=bias,
            fuse_quant=fuse_quant,
        )
    return Out.view(B, Tq, Hq, D).permute(0, 2, 1, 3).contiguous()
