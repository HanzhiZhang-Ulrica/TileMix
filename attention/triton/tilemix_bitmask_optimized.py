#!/usr/bin/env python
"""Optimized single-pass TileMix kernel that retains packed-bitmask routing.

This module is intentionally separate from :mod:`attention.triton.tilemix`.
It preserves the defining properties of Variant A:

* one packed ``uint64`` routing word per query-block/KV-head row;
* monotonically increasing K/V traversal;
* one online-softmax state and no partial-state merge;
* no routing index lists or index preprocessing.

The optimizations are independently selectable at launch:

* compile-time all-FP16 and all-INT8 specializations;
* one routing decision per packed-mask run instead of per compute tile;
* complete QK/softmax/PV updates scoped inside each precision branch;
* scalar quantization scales when BM/BN align with BLKQ/BLKK;
* an INT8 cache-policy control.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch
import triton
import triton.language as tl

from attention.triton.tilemix import (
    ALL_AUTOTUNE_CONFIGS,
    COMMON_AUTOTUNE_CONFIGS,
    LOG2E,
    quantize_qk_for_tilemix,
    quantize_qkv_for_tilemix,
)


ROUTING_MODE_MIXED = 0
ROUTING_MODE_FP16 = 1
ROUTING_MODE_INT8 = 2
RoutingMode = Literal["auto", "mixed", "fp16", "int8"]


def _copy_autotune_configs() -> list[triton.Config]:
    return [
        triton.Config(
            dict(config.kwargs),
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )
        for config in COMMON_AUTOTUNE_CONFIGS
    ]


BITMASK_OPT_AUTOTUNE_CONFIGS = _copy_autotune_configs()


@triton.jit
def _softmax_prefix_update(
    qk,
    m_i,
    l_i,
    acc,
    neg_inf_tile,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Update one FP16 online-softmax prefix and return probabilities."""

    qk_max = tl.max(qk.to(tl.float32), 1)
    tile_has_values = qk_max != float("-inf")
    m_next = tl.maximum(m_i.to(tl.float32), qk_max).to(tl.float16)
    m_next = tl.where(tile_has_values, m_next, m_i)
    qk_shifted = tl.where(
        tile_has_values[:, None],
        qk - m_next[:, None],
        neg_inf_tile,
    )
    p = tl.math.exp2(qk_shifted.to(tl.float32)).to(tl.float16)
    l_tile = tl.sum(p.to(tl.float32), 1).to(tl.float16)
    alpha = tl.where(
        tile_has_values,
        tl.math.exp2((m_i - m_next).to(tl.float32)).to(tl.float16),
        1.0,
    )
    acc *= alpha[:, None]
    l_i = (l_i * alpha + l_tile).to(tl.float16)
    return p, m_next, l_i, acc


@triton.jit
def _fp16_tile_update(
    q_fp16,
    m_i,
    l_i,
    acc,
    K,
    V,
    bias,
    k_offset,
    v_offset,
    off_h_q,
    block_n,
    offs_m,
    offs_n,
    offs_d,
    seqlen_q,
    seqlen_k,
    stride_kn,
    stride_kk,
    stride_vk,
    stride_vn,
    stride_bh,
    stride_bm,
    stride_bn,
    neg_inf_tile,
    IS_CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Process one sequential FP16-routed K/V tile."""

    start_n = block_n * BLOCK_N
    key_offsets = start_n + offs_n
    key_valid = key_offsets < seqlen_k
    k_ptrs = (
        K
        + k_offset
        + offs_d[:, None] * stride_kk
        + key_offsets[None, :] * stride_kn
    )
    k_mask = (offs_d[:, None] < ACTUAL_BLOCK_DMODEL) & key_valid[None, :]
    k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0, cache_modifier=".ca")
    qk = tl.dot(q_fp16, k_tile, out_dtype=tl.float16)

    attn_valid = key_valid[None, :]
    if IS_CAUSAL:
        causal_offset = seqlen_q - seqlen_k
        attn_valid = attn_valid & (
            offs_m[:, None] >= (key_offsets + causal_offset)[None, :]
        )
    qk = tl.where(attn_valid, qk, neg_inf_tile)

    if HAS_BIAS:
        # Match Variant A's existing varlen bias addressing.
        bias_ptrs = (
            bias
            + off_h_q * stride_bh
            + offs_m[:, None] * stride_bm
            + key_offsets[None, :] * stride_bn
        )
        b = tl.load(bias_ptrs, mask=attn_valid, other=0.0)
        qk += (b * LOG2E).to(tl.float16)

    p, m_i, l_i, acc = _softmax_prefix_update(
        qk,
        m_i,
        l_i,
        acc,
        neg_inf_tile,
        BLOCK_M,
        BLOCK_N,
    )

    v_ptrs = (
        V
        + v_offset
        + key_offsets[:, None] * stride_vk
        + offs_d[None, :] * stride_vn
    )
    v_mask = key_valid[:, None] & (
        offs_d[None, :] < ACTUAL_BLOCK_DMODEL
    )
    v_tile = tl.load(
        v_ptrs,
        mask=v_mask,
        other=0.0,
        cache_modifier=".ca",
    )
    acc += tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float16)
    return m_i, l_i, acc


@triton.jit
def _int8_tile_update(
    q8_tile,
    sq_scalar,
    sq_vec,
    qk_scale,
    m_i,
    l_i,
    acc,
    K8,
    V8,
    V,
    bias,
    sK_blocks,
    sV_blocks,
    k8_offset,
    v8_offset,
    v_offset,
    k_scale_base,
    k_scale_end,
    v_scale_base,
    v_scale_end,
    off_h_k,
    off_h_q,
    block_n,
    offs_m,
    offs_n,
    offs_d,
    seqlen_q,
    seqlen_k,
    stride_k8n,
    stride_k8k,
    stride_v8k,
    stride_v8n,
    stride_vk,
    stride_vn,
    stride_bh,
    stride_bm,
    stride_bn,
    neg_inf_tile,
    IS_CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SCALAR_SCALES: tl.constexpr,
    INT8_CACHE_CA: tl.constexpr,
    INT8_USE_FP16_V: tl.constexpr,
    BLKK: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Process one sequential INT8-routed score tile and its FP16 PV."""

    start_n = block_n * BLOCK_N
    key_offsets = start_n + offs_n
    key_valid = key_offsets < seqlen_k
    k8_ptrs = (
        K8
        + k8_offset
        + offs_d[:, None] * stride_k8k
        + key_offsets[None, :] * stride_k8n
    )
    k8_mask = (offs_d[:, None] < ACTUAL_BLOCK_DMODEL) & key_valid[None, :]
    if INT8_CACHE_CA:
        k8_tile = tl.load(
            k8_ptrs,
            mask=k8_mask,
            other=0,
            cache_modifier=".ca",
        ).to(tl.int8)
    else:
        k8_tile = tl.load(
            k8_ptrs,
            mask=k8_mask,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int8)

    if SCALAR_SCALES:
        k_scale_index = k_scale_base + start_n // BLKK
        k_scale_index = tl.minimum(k_scale_index, k_scale_end - 1)
        sk_scalar = tl.load(
            sK_blocks + k_scale_index * HK + off_h_k,
            mask=k_scale_end > k_scale_base,
            other=1.0,
        ).to(tl.float16)
        score_scale = (sq_scalar * sk_scalar * qk_scale).to(tl.float16)
    else:
        k_scale_indices = k_scale_base + key_offsets.to(tl.int32) // BLKK
        k_scale_indices = tl.minimum(k_scale_indices, k_scale_end - 1)
        sk_vec = tl.load(
            sK_blocks + k_scale_indices * HK + off_h_k,
            mask=key_valid,
            other=1.0,
        ).to(tl.float16)
        score_scale = (
            sq_vec[:, None] * sk_vec[None, :] * qk_scale
        ).to(tl.float16)

    qk_i32 = tl.dot(q8_tile, k8_tile, out_dtype=tl.int32)
    qk = (
        qk_i32.to(tl.float32) * score_scale.to(tl.float32)
    ).to(tl.float16)

    attn_valid = key_valid[None, :]
    if IS_CAUSAL:
        causal_offset = seqlen_q - seqlen_k
        attn_valid = attn_valid & (
            offs_m[:, None] >= (key_offsets + causal_offset)[None, :]
        )
    qk = tl.where(attn_valid, qk, neg_inf_tile)
    if HAS_BIAS:
        bias_ptrs = (
            bias
            + off_h_q * stride_bh
            + offs_m[:, None] * stride_bm
            + key_offsets[None, :] * stride_bn
        )
        b = tl.load(bias_ptrs, mask=attn_valid, other=0.0)
        qk += (b * LOG2E).to(tl.float16)
    p, m_i, l_i, acc = _softmax_prefix_update(
        qk,
        m_i,
        l_i,
        acc,
        neg_inf_tile,
        BLOCK_M,
        BLOCK_N,
    )

    if INT8_USE_FP16_V:
        v_ptrs = (
            V
            + v_offset
            + key_offsets[:, None] * stride_vk
            + offs_d[None, :] * stride_vn
        )
        v_mask = key_valid[:, None] & (
            offs_d[None, :] < ACTUAL_BLOCK_DMODEL
        )
        v_tile = tl.load(
            v_ptrs,
            mask=v_mask,
            other=0.0,
            cache_modifier=".ca",
        )
        pv = tl.dot(p.to(v_tile.dtype), v_tile).to(tl.float16)
    else:
        v8_ptrs = (
            V8
            + v8_offset
            + key_offsets[:, None] * stride_v8k
            + offs_d[None, :] * stride_v8n
        )
        v8_mask = key_valid[:, None] & (
            offs_d[None, :] < ACTUAL_BLOCK_DMODEL
        )
        if INT8_CACHE_CA:
            v8_tile = tl.load(
                v8_ptrs,
                mask=v8_mask,
                other=0,
                cache_modifier=".ca",
            ).to(tl.int8)
        else:
            v8_tile = tl.load(
                v8_ptrs,
                mask=v8_mask,
                other=0,
                cache_modifier=".cg",
            ).to(tl.int8)

        if SCALAR_SCALES:
            v_scale_index = v_scale_base + start_n // BLKK
            v_scale_index = tl.minimum(v_scale_index, v_scale_end - 1)
            sv_scalar = tl.load(
                sV_blocks + v_scale_index * HK + off_h_k,
                mask=v_scale_end > v_scale_base,
                other=1.0,
            ).to(tl.float16)
            p_scaled = (p * sv_scalar).to(tl.float16)
        else:
            v_scale_indices = (
                v_scale_base + key_offsets.to(tl.int32) // BLKK
            )
            v_scale_indices = tl.minimum(
                v_scale_indices,
                v_scale_end - 1,
            )
            sv_vec = tl.load(
                sV_blocks + v_scale_indices * HK + off_h_k,
                mask=key_valid,
                other=1.0,
            ).to(tl.float16)
            p_scaled = (p * sv_vec[None, :]).to(tl.float16)
        pv = tl.dot(p_scaled, v8_tile.to(tl.float16)).to(tl.float16)

    acc += pv
    return m_i, l_i, acc


@triton.autotune(
    configs=BITMASK_OPT_AUTOTUNE_CONFIGS,
    key=["BLOCK_DMODEL", "BLOCK_M", "BLOCK_N"],
)
@triton.jit
def attn_fwd_bitmask_optimized(
    Q,
    K,
    V,
    bias,
    sm_scale,
    Out,
    precision_bitmask,
    Q8,
    K8,
    V8,
    sQ_blocks,
    sK_blocks,
    sV_blocks,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_q_scale,
    cu_k_scale,
    cu_v_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_q8z,
    stride_q8h,
    stride_q8m,
    stride_q8k,
    stride_k8z,
    stride_k8h,
    stride_k8n,
    stride_k8k,
    stride_v8z,
    stride_v8h,
    stride_v8k,
    stride_v8n,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BLOCK_N_MASK: tl.constexpr,
    BLKQ: tl.constexpr,
    BLKK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ROUTING_MODE: tl.constexpr,
    RUN_LENGTH: tl.constexpr,
    SCALAR_SCALES: tl.constexpr,
    INT8_CACHE_CA: tl.constexpr,
    INT8_USE_FP16_V: tl.constexpr,
):
    """Single-state, sequential bitmask kernel with homogeneous inner runs."""

    tl.static_assert(
        BLOCK_N_MASK % BLOCK_N == 0,
        "BLOCK_N_MASK must be a multiple of BLOCK_N",
    )
    MASK_GROUP: tl.constexpr = BLOCK_N_MASK // BLOCK_N

    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    cu_q_start = tl.load(cu_seqlens_q + off_z)
    cu_q_end = tl.load(cu_seqlens_q + off_z + 1)
    cu_k_start = tl.load(cu_seqlens_k + off_z)
    cu_k_end = tl.load(cu_seqlens_k + off_z + 1)
    seqlen_q = cu_q_end - cu_q_start
    seqlen_k = cu_k_end - cu_k_start
    if start_m * BLOCK_M >= seqlen_q:
        return

    n_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        causal_n_blocks = (
            (start_m + 1) * BLOCK_M
            + seqlen_k
            - seqlen_q
            + BLOCK_N
            - 1
        ) // BLOCK_N
        n_blocks = tl.minimum(n_blocks, causal_n_blocks)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    neg_inf_tile = tl.full(
        [BLOCK_M, BLOCK_N],
        float("-inf"),
        dtype=tl.float16,
    )
    zero_acc = tl.zeros(
        [BLOCK_M, BLOCK_DMODEL],
        dtype=tl.float16,
    )

    q_offset = (
        off_z * stride_qz
        + off_h_q * stride_qh
        + cu_q_start * stride_qm
    )
    q_ptrs = (
        Q
        + q_offset
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q_valid = (offs_m[:, None] < seqlen_q) & (
        offs_d[None, :] < ACTUAL_BLOCK_DMODEL
    )
    qk_scale = sm_scale * LOG2E

    if ROUTING_MODE != 2:
        q_unscaled = tl.load(
            q_ptrs,
            mask=q_valid,
            other=0.0,
        ).to(tl.float16)
        q_fp16 = (q_unscaled * qk_scale).to(tl.float16)

    if ROUTING_MODE != 1:
        q8_offset = (
            off_z * stride_q8z
            + off_h_q * stride_q8h
            + cu_q_start * stride_q8m
        )
        q8_ptrs = (
            Q8
            + q8_offset
            + offs_m[:, None] * stride_q8m
            + offs_d[None, :] * stride_q8k
        )
        q8_tile = tl.load(
            q8_ptrs,
            mask=q_valid,
            other=0,
        ).to(tl.int8)

        q_scale_base = tl.load(cu_q_scale + off_z)
        q_scale_end = tl.load(cu_q_scale + off_z + 1)
        if SCALAR_SCALES:
            q_scale_index = (
                q_scale_base + start_m * BLOCK_M // BLKQ
            )
            q_scale_index = tl.minimum(q_scale_index, q_scale_end - 1)
            sq_scalar = tl.load(
                sQ_blocks + q_scale_index * HQ + off_h_q,
                mask=q_scale_end > q_scale_base,
                other=1.0,
            ).to(tl.float16)
            sq_vec = tl.zeros([BLOCK_M], dtype=tl.float16)
        else:
            q_scale_indices = (
                q_scale_base + offs_m.to(tl.int32) // BLKQ
            )
            q_scale_indices = tl.minimum(
                q_scale_indices,
                q_scale_end - 1,
            )
            sq_vec = tl.load(
                sQ_blocks + q_scale_indices * HQ + off_h_q,
                mask=offs_m < seqlen_q,
                other=1.0,
            ).to(tl.float16)
            sq_scalar = tl.full([], 1.0, dtype=tl.float16)

        k_scale_base = tl.load(cu_k_scale + off_z)
        k_scale_end = tl.load(cu_k_scale + off_z + 1)
        if INT8_USE_FP16_V:
            v_scale_base = tl.full([], 0, dtype=tl.int32)
            v_scale_end = tl.full([], 0, dtype=tl.int32)
        else:
            v_scale_base = tl.load(cu_v_scale + off_z)
            v_scale_end = tl.load(cu_v_scale + off_z + 1)

    k_offset = (
        off_z * stride_kz
        + off_h_k * stride_kh
        + cu_k_start * stride_kn
    )
    v_offset = (
        off_z * stride_vz
        + off_h_k * stride_vh
        + cu_k_start * stride_vk
    )
    k8_offset = (
        off_z * stride_k8z
        + off_h_k * stride_k8h
        + cu_k_start * stride_k8n
    )
    v8_offset = (
        off_z * stride_v8z
        + off_h_k * stride_v8h
        + cu_k_start * stride_v8k
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float16)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float16)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float16)

    if ROUTING_MODE == 1:
        for block_n in tl.range(0, n_blocks, 1):
            m_i, l_i, acc = _fp16_tile_update(
                q_fp16,
                m_i,
                l_i,
                acc,
                K,
                V,
                bias,
                k_offset,
                v_offset,
                off_h_q,
                block_n,
                offs_m,
                offs_n,
                offs_d,
                seqlen_q,
                seqlen_k,
                stride_kn,
                stride_kk,
                stride_vk,
                stride_vn,
                stride_bh,
                stride_bm,
                stride_bn,
                neg_inf_tile,
                IS_CAUSAL,
                HAS_BIAS,
                ACTUAL_BLOCK_DMODEL,
                BLOCK_M,
                BLOCK_N,
                BLOCK_DMODEL,
            )
    elif ROUTING_MODE == 2:
        for block_n in tl.range(0, n_blocks, 1):
            m_i, l_i, acc = _int8_tile_update(
                q8_tile,
                sq_scalar,
                sq_vec,
                qk_scale,
                m_i,
                l_i,
                acc,
                K8,
                V8,
                V,
                bias,
                sK_blocks,
                sV_blocks,
                k8_offset,
                v8_offset,
                v_offset,
                k_scale_base,
                k_scale_end,
                v_scale_base,
                v_scale_end,
                off_h_k,
                off_h_q,
                block_n,
                offs_m,
                offs_n,
                offs_d,
                seqlen_q,
                seqlen_k,
                stride_k8n,
                stride_k8k,
                stride_v8k,
                stride_v8n,
                stride_vk,
                stride_vn,
                stride_bh,
                stride_bm,
                stride_bn,
                neg_inf_tile,
                IS_CAUSAL,
                HAS_BIAS,
                SCALAR_SCALES,
                INT8_CACHE_CA,
                INT8_USE_FP16_V,
                BLKK,
                HK,
                ACTUAL_BLOCK_DMODEL,
                BLOCK_M,
                BLOCK_N,
                BLOCK_DMODEL,
            )
    else:
        m_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
        routing_row = off_h_k * m_blocks + start_m
        row_bitmask = tl.load(
            precision_bitmask + routing_row
        ).to(tl.uint64)
        route_blocks = (n_blocks + MASK_GROUP - 1) // MASK_GROUP
        route_pos = tl.full([], 0, dtype=tl.int32)

        while route_pos < route_blocks:
            route_shift = route_pos.to(tl.uint64)
            use_int8 = (
                (row_bitmask >> route_shift) & 1
            ).to(tl.int32)
            run_end = route_pos + 1
            if RUN_LENGTH:
                searching = tl.full([], 1, dtype=tl.int32)
                while (run_end < route_blocks) & (searching != 0):
                    next_shift = run_end.to(tl.uint64)
                    next_int8 = (
                        (row_bitmask >> next_shift) & 1
                    ).to(tl.int32)
                    same_route = next_int8 == use_int8
                    run_end += same_route.to(tl.int32)
                    searching = same_route.to(tl.int32)

            first_block = route_pos * MASK_GROUP
            last_block = tl.minimum(run_end * MASK_GROUP, n_blocks)
            if use_int8 != 0:
                for block_n in tl.range(first_block, last_block, 1):
                    m_i, l_i, acc = _int8_tile_update(
                        q8_tile,
                        sq_scalar,
                        sq_vec,
                        qk_scale,
                        m_i,
                        l_i,
                        acc,
                        K8,
                        V8,
                        V,
                        bias,
                        sK_blocks,
                        sV_blocks,
                        k8_offset,
                        v8_offset,
                        v_offset,
                        k_scale_base,
                        k_scale_end,
                        v_scale_base,
                        v_scale_end,
                        off_h_k,
                        off_h_q,
                        block_n,
                        offs_m,
                        offs_n,
                        offs_d,
                        seqlen_q,
                        seqlen_k,
                        stride_k8n,
                        stride_k8k,
                        stride_v8k,
                        stride_v8n,
                        stride_vk,
                        stride_vn,
                        stride_bh,
                        stride_bm,
                        stride_bn,
                        neg_inf_tile,
                        IS_CAUSAL,
                        HAS_BIAS,
                        SCALAR_SCALES,
                        INT8_CACHE_CA,
                        INT8_USE_FP16_V,
                        BLKK,
                        HK,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                    )
            else:
                for block_n in tl.range(first_block, last_block, 1):
                    m_i, l_i, acc = _fp16_tile_update(
                        q_fp16,
                        m_i,
                        l_i,
                        acc,
                        K,
                        V,
                        bias,
                        k_offset,
                        v_offset,
                        off_h_q,
                        block_n,
                        offs_m,
                        offs_n,
                        offs_d,
                        seqlen_q,
                        seqlen_k,
                        stride_kn,
                        stride_kk,
                        stride_vk,
                        stride_vn,
                        stride_bh,
                        stride_bm,
                        stride_bn,
                        neg_inf_tile,
                        IS_CAUSAL,
                        HAS_BIAS,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                    )
            route_pos = run_end

    acc /= l_i[:, None]
    acc = acc.to(Out.type.element_ty)
    o_offset = (
        off_z * stride_oz
        + off_h_q * stride_oh
        + cu_q_start * stride_om
    )
    o_ptrs = (
        Out
        + o_offset
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    o_valid = (offs_m[:, None] < seqlen_q) & (
        offs_d[None, :] < ACTUAL_BLOCK_DMODEL
    )
    tl.store(o_ptrs, tl.where(o_valid, acc, zero_acc), mask=o_valid)


def _routing_mode_code(mode: str) -> int:
    normalized = mode.strip().lower()
    codes = {
        "mixed": ROUTING_MODE_MIXED,
        "fp16": ROUTING_MODE_FP16,
        "int8": ROUTING_MODE_INT8,
    }
    if normalized not in codes:
        raise ValueError(
            f"routing_mode must be one of {tuple(codes)}, got {mode!r}"
        )
    return codes[normalized]


def classify_bitmask_routing(
    precision_bitmask: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    block_m: int,
    block_n: int,
    block_n_mask: int,
    causal: bool,
) -> Literal["mixed", "fp16", "int8"]:
    """Classify globally homogeneous routing without creating index lists."""

    if block_n_mask % block_n != 0:
        raise ValueError("block_n_mask must be a multiple of block_n")
    m_blocks = (seqlen_q + block_m - 1) // block_m
    compute_blocks = (seqlen_k + block_n - 1) // block_n
    mask_group = block_n_mask // block_n
    host_masks = precision_bitmask.detach().reshape(-1).cpu().tolist()
    if len(host_masks) == 0 or len(host_masks) % m_blocks:
        raise ValueError("precision_bitmask has an invalid row count")

    all_fp16 = True
    all_int8 = True
    for row, raw_mask in enumerate(host_masks):
        query_block = row % m_blocks
        row_blocks = compute_blocks
        if causal:
            numerator = (
                (query_block + 1) * block_m + seqlen_k - seqlen_q
            )
            row_blocks = max(
                0,
                min(
                    compute_blocks,
                    (numerator + block_n - 1) // block_n,
                ),
            )
        route_blocks = (row_blocks + mask_group - 1) // mask_group
        valid_mask = (1 << route_blocks) - 1 if route_blocks else 0
        routed = int(raw_mask) & valid_mask
        all_fp16 &= routed == 0
        all_int8 &= routed == valid_mask
        if not all_fp16 and not all_int8:
            return "mixed"
    if all_fp16:
        return "fp16"
    if all_int8:
        return "int8"
    return "mixed"


def _matching_configs(
    block_m: int,
    block_n: int,
    max_seqlens_q: int,
    max_seqlens_k: int,
    compute_capability: tuple[int, int],
) -> list[triton.Config]:
    forced = os.environ.get("TILEMIX_FORCE_CONFIG")
    if forced:
        parts = [part.strip() for part in forced.split(",")]
        forced_m = int(parts[0])
        forced_n = int(parts[1])
        if (forced_m, forced_n) != (block_m, block_n):
            raise ValueError(
                "TILEMIX_FORCE_CONFIG block sizes must match the optimized "
                f"kernel request: forced=({forced_m}, {forced_n}), "
                f"requested=({block_m}, {block_n})"
            )
        warps = int(parts[2]) if len(parts) > 2 and parts[2] else 4
        stages = int(parts[3]) if len(parts) > 3 and parts[3] else 2
        return [
                triton.Config(
                {
                    "BLOCK_M": forced_m,
                    "BLOCK_N": forced_n,
                    "PRE_LOAD_V": False,
                },
                num_stages=stages,
                num_warps=warps,
            )
        ]

    legal = []
    for config in ALL_AUTOTUNE_CONFIGS:
        if config.kwargs.get("BLOCK_M") != block_m:
            continue
        if config.kwargs.get("BLOCK_N") != block_n:
            continue
        if block_m > max_seqlens_q or block_n > max_seqlens_k:
            continue
        if config.kwargs.get("PRE_LOAD_V", False):
            continue
        if compute_capability[0] < 9 and config.num_warps > 8:
            continue
        legal.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "PRE_LOAD_V": False,
                },
                num_stages=config.num_stages,
                num_warps=config.num_warps,
            )
        )
    if legal:
        return legal
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "PRE_LOAD_V": False,
            },
            num_stages=2,
            num_warps=4,
        )
    ]


def _dummy_quantization_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Correctly ranked placeholders for the compile-time FP16 fast path."""

    q8 = torch.empty(
        (1, q.shape[1], q.shape[2]),
        dtype=torch.int8,
        device=q.device,
    )
    k8 = torch.empty(
        (1, k.shape[1], k.shape[2]),
        dtype=torch.int8,
        device=k.device,
    )
    v8 = torch.empty(
        (1, v.shape[1], v.shape[2]),
        dtype=torch.int8,
        device=v.device,
    )
    sq = torch.empty((1, q.shape[1]), dtype=q.dtype, device=q.device)
    sk = torch.empty((1, k.shape[1]), dtype=k.dtype, device=k.device)
    sv = torch.empty_like(sk)
    cuq = torch.zeros_like(cu_seqlens_q, dtype=torch.int32)
    cuk = torch.zeros_like(cu_seqlens_k, dtype=torch.int32)
    cuv = torch.zeros_like(cu_seqlens_k, dtype=torch.int32)
    return q8, sq, cuq, k8, sk, cuk, v8, sv, cuv


def _dummy_v_quantization_metadata(
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimal placeholders removed by the compile-time FP16-PV path."""

    v8 = torch.empty(
        (1, v.shape[1], v.shape[2]),
        dtype=torch.int8,
        device=v.device,
    )
    sv = torch.empty((1, v.shape[1]), dtype=v.dtype, device=v.device)
    cuv = torch.zeros_like(cu_seqlens_k, dtype=torch.int32)
    return v8, sv, cuv


@torch.no_grad()
def launch_bitmask_optimized_prequantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlens_q: int,
    max_seqlens_k: int,
    precision_bitmask: torch.Tensor,
    *,
    q8: torch.Tensor,
    k8: torch.Tensor,
    v8: torch.Tensor,
    sQ_blocks: torch.Tensor,
    sK_blocks: torch.Tensor,
    sV_blocks: torch.Tensor,
    cu_q_scale: torch.Tensor,
    cu_k_scale: torch.Tensor,
    cu_v_scale: torch.Tensor,
    block_m: int = 128,
    block_n: int = 64,
    block_n_mask: Optional[int] = None,
    BLKQ: int = 128,
    BLKK: int = 64,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    routing_mode: RoutingMode = "auto",
    run_length: bool = False,
    scalar_scales: bool = True,
    int8_cache_ca: bool = False,
    int8_use_fp16_v: bool = False,
) -> tuple[torch.Tensor, Literal["mixed", "fp16", "int8"]]:
    """Launch the no-index optimized kernel with quantization excluded."""

    if not q.is_cuda:
        raise RuntimeError("optimized TileMix is a CUDA-only Triton kernel")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected flattened [tokens, heads, dim] tensors")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("K/V shapes and Q/K/V head dimensions must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must share one CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must share one dtype")
    if max_seqlens_q != max_seqlens_k:
        raise ValueError("packed routing currently requires equal Q/K lengths")
    if block_n_mask is None:
        block_n_mask = block_n
    if block_n_mask % block_n:
        raise ValueError("block_n_mask must be a multiple of block_n")
    if (max_seqlens_k + block_n_mask - 1) // block_n_mask > 64:
        raise ValueError("packed routing exceeds 64 mask blocks")
    if scalar_scales and (block_m != BLKQ or block_n != BLKK):
        raise ValueError(
            "scalar_scales requires block_m == BLKQ and block_n == BLKK"
        )

    expected_rows = k.shape[1] * (
        (max_seqlens_q + block_m - 1) // block_m
    )
    if precision_bitmask.dtype != torch.uint64:
        raise TypeError("precision_bitmask must be torch.uint64")
    if precision_bitmask.numel() != expected_rows:
        raise ValueError(
            f"precision_bitmask has {precision_bitmask.numel()} rows, "
            f"expected {expected_rows}"
        )

    normalized_mode = routing_mode.strip().lower()
    if normalized_mode == "auto":
        normalized_mode = classify_bitmask_routing(
            precision_bitmask,
            seqlen_q=max_seqlens_q,
            seqlen_k=max_seqlens_k,
            block_m=block_m,
            block_n=block_n,
            block_n_mask=block_n_mask,
            causal=causal,
        )
    mode_code = _routing_mode_code(normalized_mode)

    quant_tensors = (
        q8,
        k8,
        v8,
        sQ_blocks,
        sK_blocks,
        sV_blocks,
        cu_q_scale,
        cu_k_scale,
        cu_v_scale,
    )
    if any(t.device != q.device for t in quant_tensors):
        raise ValueError("all quantization tensors must share q.device")
    if q8.dtype != torch.int8 or k8.dtype != torch.int8:
        raise TypeError("q8 and k8 must be torch.int8")
    if v8.dtype != torch.int8:
        raise TypeError("v8 must be torch.int8")

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    if o is None:
        o = torch.empty_like(q)
    cu_seqlens_q = cu_seqlens_q.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()
    precision_bitmask = precision_bitmask.to(
        device=q.device,
        dtype=torch.uint64,
    ).contiguous()

    _, heads_q, head_dim = q.shape
    _, heads_k, _ = k.shape
    batch = cu_seqlens_q.numel() - 1
    padded_head_dim = max(32, 1 << (head_dim - 1).bit_length())
    padded_head_dim = ((padded_head_dim + 31) // 32) * 32
    if padded_head_dim > 128:
        raise ValueError("head dimension must be <= 128")

    q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
    k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
    v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
    o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    q8_strides = (0, q8.stride(1), q8.stride(0), q8.stride(2))
    k8_strides = (0, k8.stride(1), k8.stride(0), k8.stride(2))
    v8_strides = (0, v8.stride(1), v8.stride(0), v8.stride(2))
    bias_strides = (
        (0, 0, 0, 0)
        if bias is None
        else (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )
    )

    compute_capability = torch.cuda.get_device_capability(q.device)
    attn_fwd_bitmask_optimized.configs = _matching_configs(
        block_m,
        block_n,
        max_seqlens_q,
        max_seqlens_k,
        compute_capability,
    )

    def grid(meta):
        return (
            triton.cdiv(max_seqlens_q, meta["BLOCK_M"]),
            heads_q,
            batch,
        )

    attn_fwd_bitmask_optimized[grid](
        q,
        k,
        v,
        bias,
        sm_scale,
        o,
        precision_bitmask,
        q8,
        k8,
        v8,
        sQ_blocks,
        sK_blocks,
        sV_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_q_scale,
        cu_k_scale,
        cu_v_scale,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *q8_strides,
        *k8_strides,
        *v8_strides,
        HQ=heads_q,
        HK=heads_k,
        ACTUAL_BLOCK_DMODEL=head_dim,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        BLOCK_DMODEL=padded_head_dim,
        BLOCK_N_MASK=block_n_mask,
        BLKQ=BLKQ,
        BLKK=BLKK,
        HAS_BIAS=bias is not None,
        ROUTING_MODE=mode_code,
        RUN_LENGTH=run_length,
        SCALAR_SCALES=scalar_scales,
        INT8_CACHE_CA=int8_cache_ca,
        INT8_USE_FP16_V=int8_use_fp16_v,
    )
    return o, normalized_mode


@torch.no_grad()
def tilemix_attention_bitmask_optimized_with_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlens_q: int,
    max_seqlens_k: int,
    precision_bitmask: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 64,
    block_n_mask: Optional[int] = None,
    BLKQ: int = 128,
    BLKK: int = 64,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    routing_mode: RoutingMode = "auto",
    run_length: bool = False,
    scalar_scales: bool = True,
    int8_cache_ca: bool = False,
    int8_use_fp16_v: bool = False,
) -> tuple[torch.Tensor, Literal["mixed", "fp16", "int8"]]:
    """Complete optimized invocation; homogeneous FP16 skips quantization."""

    if block_n_mask is None:
        block_n_mask = block_n
    normalized_mode = routing_mode.strip().lower()
    if normalized_mode == "auto":
        normalized_mode = classify_bitmask_routing(
            precision_bitmask,
            seqlen_q=max_seqlens_q,
            seqlen_k=max_seqlens_k,
            block_m=block_m,
            block_n=block_n,
            block_n_mask=block_n_mask,
            causal=causal,
        )

    if normalized_mode == "fp16":
        quantized = _dummy_quantization_metadata(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
        )
    elif int8_use_fp16_v:
        q8, sq, cuq, k8, sk, cuk = quantize_qk_for_tilemix(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlens_q,
            max_seqlens_k,
            BLKQ=BLKQ,
            BLKK=BLKK,
            sm_scale=sm_scale,
        )
        v8, sv, cuv = _dummy_v_quantization_metadata(v, cu_seqlens_k)
        quantized = q8, sq, cuq, k8, sk, cuk, v8, sv, cuv
    else:
        quantized = quantize_qkv_for_tilemix(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlens_q,
            max_seqlens_k,
            BLKQ=BLKQ,
            BLKK=BLKK,
            sm_scale=sm_scale,
        )

    q8, sq, cuq, k8, sk, cuk, v8, sv, cuv = quantized
    return launch_bitmask_optimized_prequantized(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        precision_bitmask,
        q8=q8,
        k8=k8,
        v8=v8,
        sQ_blocks=sq,
        sK_blocks=sk,
        sV_blocks=sv,
        cu_q_scale=cuq,
        cu_k_scale=cuk,
        cu_v_scale=cuv,
        block_m=block_m,
        block_n=block_n,
        block_n_mask=block_n_mask,
        BLKQ=BLKQ,
        BLKK=BLKK,
        causal=causal,
        sm_scale=sm_scale,
        bias=bias,
        routing_mode=normalized_mode,
        run_length=run_length,
        scalar_scales=scalar_scales,
        int8_cache_ca=int8_cache_ca,
        int8_use_fp16_v=int8_use_fp16_v,
    )
