#!/usr/bin/env python

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
try:
    from triton.runtime.errors import OutOfResources
except ImportError:
    OutOfResources = type('OutOfResources', (RuntimeError,), {})

from attention.triton.quant_per_block_varlen import (
    per_block_int8 as quant_per_block_int8,
    per_block_int8_single,
)

LOG2E = tl.constexpr(1.4426950408889634)

COMMON_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=1, num_warps=2),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=2, num_warps=2),

    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=3, num_warps=4),

    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":True }, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":True }, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":64,  "PRE_LOAD_V":True }, num_stages=3, num_warps=4),

    triton.Config({"BLOCK_M":64,  "BLOCK_N":128, "PRE_LOAD_V":False}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":128, "PRE_LOAD_V":False}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M":64,  "BLOCK_N":128, "PRE_LOAD_V":True }, num_stages=1, num_warps=4),

    triton.Config({"BLOCK_M":128, "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=1, num_warps=4),
    triton.Config({"BLOCK_M":128, "BLOCK_N":64,  "PRE_LOAD_V":False}, num_stages=2, num_warps=4),

    triton.Config({"BLOCK_M":64,  "BLOCK_N":256, "PRE_LOAD_V":False}, num_stages=1, num_warps=4),
]



ALL_AUTOTUNE_CONFIGS = tuple(COMMON_AUTOTUNE_CONFIGS)

_AUTOTUNE_CONFIG_CACHE = {}

def _get_filtered_configs(
    max_seqlens_q: int,
    max_seqlens_k: int,
    compute_capability: tuple[int, int] | None,
):
    cc_major = compute_capability[0] if compute_capability is not None else None
    key = (max_seqlens_q, max_seqlens_k, cc_major)
    cached = _AUTOTUNE_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    filtered = list(
        _filter_autotune_configs(
            max_seqlens_q=max_seqlens_q,
            max_seqlens_k=max_seqlens_k,
            compute_capability=compute_capability,
        )
    )
    _AUTOTUNE_CONFIG_CACHE[key] = filtered
    return filtered


def _filter_autotune_configs(
    max_seqlens_q: int,
    max_seqlens_k: int,
    compute_capability: tuple[int, int] | None,
) -> list[triton.Config]:
    """
    Select runtime-valid autotune configs based on sequence limits and device arch.
    """

    cc_major = compute_capability[0] if compute_capability is not None else None

    def is_legal(cfg: triton.Config) -> bool:
        kwargs = cfg.kwargs
        if kwargs["BLOCK_M"] > max_seqlens_q:
            return False
        if kwargs["BLOCK_N"] > max_seqlens_k:
            return False
        if cc_major is not None and cc_major < 8 and kwargs.get("PRE_LOAD_V", False):
            return False
        if cc_major is not None and cc_major < 9 and getattr(cfg, "num_warps", 8) > 8:
            return False
        bm = kwargs.get("BLOCK_M", 0)
        bn = kwargs.get("BLOCK_N", 0)
        stages = getattr(cfg, "num_stages", 2)
        warps = getattr(cfg, "num_warps", 8)
        
        seqlen_factor = max_seqlens_k / 1024.0
        
        block_size_score = bm * bn
        stage_penalty = stages * 1.5
        warp_penalty = max(1.0, warps / 8.0)
        
        memory_pressure = block_size_score * stage_penalty * warp_penalty * (1.0 + seqlen_factor * 0.1)
        
        pressure_threshold = 50000.0 / (1.0 + seqlen_factor * 0.2)
        
        if memory_pressure > pressure_threshold:
            return False
        
        if bm >= 256 and bn >= 128:
            max_stages = max(1, int(3.0 / (1.0 + seqlen_factor * 0.1)))
            if stages > max_stages:
                return False
        if bm >= 256 and bn >= 64:
            max_stages = max(2, int(4.0 / (1.0 + seqlen_factor * 0.1)))
            if stages > max_stages:
                return False
        
        return True

    legal = [cfg for cfg in ALL_AUTOTUNE_CONFIGS if is_legal(cfg)]

    if legal:
        legal.sort(
            key=lambda c: (
                c.kwargs.get("BLOCK_N", 0),
                c.kwargs.get("BLOCK_M", 0),
                getattr(c, "num_warps", 0),
                getattr(c, "num_stages", 0),
                bool(c.kwargs.get("PRE_LOAD_V", False)),
            ),
            reverse=True,
        )
        return legal[:12]

    seq_only = [
        cfg
        for cfg in ALL_AUTOTUNE_CONFIGS
        if cfg.kwargs["BLOCK_M"] <= max_seqlens_q and cfg.kwargs["BLOCK_N"] <= max_seqlens_k
    ]
    if seq_only:
        return seq_only[:8]

    return [ALL_AUTOTUNE_CONFIGS[-1]]


COMMON_AUTOTUNE_KEY = [
    "BLOCK_DMODEL",
    "BLOCK_M",
    "BLOCK_N",
]


@triton.autotune(
    configs=COMMON_AUTOTUNE_CONFIGS,
    key=COMMON_AUTOTUNE_KEY,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    sm_scale,
    L,
    Out,
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
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    precision_bitmask,
    Q8,
    K8,
    V8,
    sQ_blocks,
    sK_blocks,
    sV_blocks,
    cu_q_scale,
    cu_k_scale,
    cu_v_scale,
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
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    BLKQ: tl.constexpr,
    BLKK: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    FUSE_QUANT: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_DMODEL: tl.constexpr,
    BLOCK_N_MASK: tl.constexpr,
):

    LAST_M_IDX: tl.constexpr = BLOCK_M - 1
    LAST_N_IDX: tl.constexpr = BLOCK_N - 1
    tl.static_assert(BLOCK_N_MASK % BLOCK_N == 0, "BLOCK_N_MASK must be multiple of BLOCK_N")
    MASK_GROUP: tl.constexpr = BLOCK_N_MASK // BLOCK_N

    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    if (BLOCK_M > MAX_SEQLENS_Q) | (BLOCK_N > MAX_SEQLENS_K):
        return

    n_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        n_blocks_seqlen = ((start_m + 1) * BLOCK_M +
                           seqlen_k - seqlen_q + BLOCK_N - 1) // BLOCK_N
        n_blocks = tl.minimum(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            o_offset = (
                off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
            )
            O_block_ptr = tl.make_block_ptr(
                base=Out + o_offset,
                shape=(seqlen_q, BLOCK_DMODEL),
                strides=(stride_om, stride_on),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_DMODEL),
                order=(1, 0),
            )
            return

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

    q_offset = off_z * stride_qz + off_h_q * \
        stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_offset = off_z * stride_kz + off_h_k * \
        stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    v_offset = off_z * stride_vz + off_h_k * \
        stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.multiple_of(Q_block_ptr, 32)
    tl.multiple_of(K_block_ptr, 32)
    tl.multiple_of(V_block_ptr, 32)
    q8_offset = off_z * stride_q8z + off_h_q * \
        stride_q8h + cu_seqlens_q_start * stride_q8m
    Q8_block_ptr = tl.make_block_ptr(
        base=Q8 + q8_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_q8m, stride_q8k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    k8_offset = off_z * stride_k8z + off_h_k * \
        stride_k8h + cu_seqlens_k_start * stride_k8n
    K8_block_ptr = tl.make_block_ptr(
        base=K8 + k8_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_k8k, stride_k8n),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    v8_offset = off_z * stride_v8z + off_h_k * \
        stride_v8h + cu_seqlens_k_start * stride_v8k
    V8_block_ptr = tl.make_block_ptr(
        base=V8 + v8_offset,
        shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
        strides=(stride_v8k, stride_v8n),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    tl.multiple_of(Q8_block_ptr, 32)
    tl.multiple_of(K8_block_ptr, 32)
    tl.multiple_of(V8_block_ptr, 32)

    if BIAS_TYPE != 0:
        bias_ptr = tl.make_block_ptr(
            base=bias + off_h_q * stride_bh,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        bias_ptr = None
    if ENABLE_DROPOUT:
        batch_philox_offset = (
            philox_offset_base + (off_z * HQ + off_h_q) * seqlen_q * seqlen_k
        )
    else:
        batch_philox_offset = 0

    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
            base=encoded_softmax + off_h_q * seqlen_q * seqlen_k,
            shape=(seqlen_q, seqlen_k),
            strides=(seqlen_k, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        encoded_softmax_block_ptr = 0

    acc_dtype = tl.float16
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=acc_dtype)
    l_i = tl.full([BLOCK_M], 1.0, dtype=acc_dtype)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=acc_dtype)
    qk_scale = sm_scale * LOG2E
    

    v_loaded = False

    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = tl.minimum(masked_blocks, n_blocks)
    m_blocks_actual = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    n_blocks_actual = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    is_last_row = start_m == (m_blocks_actual - 1)

    row_idx = off_h_k * m_blocks_actual + start_m
    row_bitmask = tl.load(precision_bitmask + row_idx).to(tl.uint64)

    n_mask_blocks = (seqlen_k + BLOCK_N_MASK - 1) // BLOCK_N_MASK
    max_uint64: tl.constexpr = 0xFFFFFFFFFFFFFFFF
    valid_mask = tl.where(
        n_mask_blocks == 64,
        tl.full([], max_uint64, dtype=tl.uint64),
        ((1 << n_mask_blocks) - 1).to(tl.uint64),
    )
    row_bitmask = row_bitmask & valid_mask
    row_can_use_int8 = row_bitmask != 0

    q = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=acc_dtype)
    if EVEN_M & EVEN_DMODEL:
        q = tl.load(Q_block_ptr)
    else:
        if PADDED_HEAD:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q_unscaled = q
    q = (q * qk_scale).to(acc_dtype)
    
    q8_tile = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.int8)
    sq_vec = tl.zeros([BLOCK_M], dtype=acc_dtype)
    if row_can_use_int8 and not FUSE_QUANT:
        if EVEN_M & EVEN_DMODEL:
            q8_tile = tl.load(Q8_block_ptr).to(tl.int8)
        else:
            if PADDED_HEAD:
                q8_tile = tl.load(
                    Q8_block_ptr, boundary_check=(0, 1)
                ).to(tl.int8)
            else:
                q8_tile = tl.load(
                    Q8_block_ptr, boundary_check=(0,)
                ).to(tl.int8)
        q_block_base = tl.load(cu_q_scale + off_z)
        q_block_bound = tl.load(cu_q_scale + off_z + 1)
        row_offsets = offs_m.to(tl.int32)
        q_row_valid = row_offsets < seqlen_q
        q_block_idx = q_block_base + (row_offsets // BLKQ)
        q_block_idx = tl.minimum(q_block_idx, q_block_bound - 1)
        sq_ptrs = sQ_blocks + q_block_idx * HQ + off_h_q
        sq_vec = tl.load(
            sq_ptrs,
            mask=q_row_valid,
            other=1.0).to(acc_dtype)

    k_block_base = tl.load(cu_k_scale + off_z)
    k_block_bound = tl.load(cu_k_scale + off_z + 1)
    v_block_base = tl.load(cu_v_scale + off_z)
    v_block_bound = tl.load(cu_v_scale + off_z + 1)
    
    K_ptr = K_block_ptr
    V_ptr = V_block_ptr
    bias_ptr_iter = bias_ptr
    encoded_ptr = encoded_softmax_block_ptr
    K8_ptr = K8_block_ptr
    V8_ptr = V8_block_ptr

    tail_threshold = n_blocks * BLOCK_N
    n_full_blocks = n_blocks - masked_blocks
    last_full_block = n_full_blocks - 1
    if IS_CAUSAL:
        offs_n_causal = offs_n + (seqlen_q - seqlen_k)
    neg_inf_block = tl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=acc_dtype)
    zero_block = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)
    zero_block_dm = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=acc_dtype)
    
    v_preload = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=acc_dtype)

    for block_n in tl.range(0, n_blocks, 1, warp_specialize=WARP_SPECIALIZE):
        start_n = block_n * BLOCK_N
        end_n = start_n + BLOCK_N

        is_last_k_tile = block_n == (n_blocks - 1)
        
        block_n_int32 = block_n.to(tl.int32)
        
        is_last_col = block_n_int32 == (n_blocks_actual - 1)

        boundary_check_needed = is_last_k_tile & (end_n > seqlen_k)
        use_fallback = is_last_row | is_last_col
        
        mask_block_idx = (block_n_int32 // MASK_GROUP).to(tl.uint64)
        bit = ((row_bitmask >> mask_block_idx) & 1).to(tl.int32)

        prec_code = tl.where(use_fallback & (bit == 0), 0, bit)
        prec_is_int8 = prec_code != 0
        use_int8 = prec_is_int8 & row_can_use_int8

        is_tail = block_n >= n_full_blocks

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_dtype)

        if is_tail:
            if (n_extra_tokens != 0) & (end_n >= tail_threshold):
                boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
                size_n = start_n + offs_n[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, neg_inf_block)
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= (
                    start_n + offs_n_causal)[None, :]
                qk = tl.where(causal_mask, qk, neg_inf_block)

        v8_tile = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.int8)
        sv_vec = tl.zeros((BLOCK_N,), dtype=acc_dtype)

        if use_int8 and not FUSE_QUANT:
            col_offsets = (start_n + offs_n).to(tl.int32)
            col_valid = col_offsets < seqlen_k
            k_block_idx = k_block_base + (col_offsets // BLKK)
            k_block_idx = tl.minimum(k_block_idx, k_block_bound - 1)

            sk_ptrs = sK_blocks + k_block_idx * HK + off_h_k
            sk_vec = tl.load(
                sk_ptrs,
                mask=col_valid,
                other=1.0).to(acc_dtype)

            v_block_idx = v_block_base + (col_offsets // BLKK)
            v_block_idx = tl.minimum(v_block_idx, v_block_bound - 1)

            sv_ptrs = sV_blocks + v_block_idx * HK + off_h_k
            sv_vec = tl.load(
                sv_ptrs,
                mask=col_valid,
                other=1.0).to(acc_dtype)

            if boundary_check_needed:
                if PADDED_HEAD:
                    k8_loaded = tl.load(
                        K8_ptr,
                        boundary_check=(0, 1),
                        cache_modifier=".cg",
                    )
                    v8_loaded = tl.load(
                        V8_ptr,
                        boundary_check=(0, 1),
                        cache_modifier=".cg",
                    )
                else:
                    k8_loaded = tl.load(
                        K8_ptr,
                        boundary_check=(1,),
                        cache_modifier=".cg",
                    )
                    v8_loaded = tl.load(
                        V8_ptr,
                        boundary_check=(0,),
                        cache_modifier=".cg",
                    )
            else:
                if PADDED_HEAD:
                    k8_loaded = tl.load(
                        K8_ptr,
                        boundary_check=(0,),
                        cache_modifier=".cg",
                    )
                    v8_loaded = tl.load(
                        V8_ptr,
                        boundary_check=(1,),
                        cache_modifier=".cg",
                    )
                else:
                    k8_loaded = tl.load(
                        K8_ptr,
                        cache_modifier=".cg",
                    )
                    v8_loaded = tl.load(
                        V8_ptr,
                        cache_modifier=".cg",
                    )

            k8_tile = k8_loaded.to(tl.int8)
            v8_tile = v8_loaded.to(tl.int8)

            qk_i32 = tl.dot(q8_tile, k8_tile, out_dtype=tl.int32)
            scale = (sq_vec[:, None] * sk_vec[None, :]).to(acc_dtype)
            scale = (scale * qk_scale).to(acc_dtype)
            # Apply the dequantization scale before narrowing to FP16.
            # Raw INT32 dot products can exceed 65504 at long contexts.
            qk += (qk_i32.to(tl.float32) * scale.to(tl.float32)).to(acc_dtype)
        elif not use_int8 or FUSE_QUANT:
            if boundary_check_needed:
                if PADDED_HEAD:
                    k = tl.load(
                        K_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        cache_modifier=".ca",
                    )
                else:
                    k = tl.load(
                        K_ptr,
                        boundary_check=(1,),
                        padding_option="zero",
                        cache_modifier=".ca",
                    )
            else:
                if PADDED_HEAD:
                    k = tl.load(
                        K_ptr,
                        boundary_check=(0,),
                        padding_option="zero",
                        cache_modifier=".ca",
                    )
                else:
                    k = tl.load(K_ptr, cache_modifier=".ca")
            
            if PRE_LOAD_V and not v_loaded:
                if boundary_check_needed:
                    if PADDED_HEAD:
                        v_preload = tl.load(
                            V_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                    else:
                        v_preload = tl.load(
                            V_ptr,
                            boundary_check=(0,),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                else:
                    if PADDED_HEAD:
                        v_preload = tl.load(
                            V_ptr,
                            boundary_check=(1,),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                    else:
                        v_preload = tl.load(V_ptr, cache_modifier=".ca")
                v_loaded = True
            
            if FUSE_QUANT and use_int8 and row_can_use_int8 and prec_is_int8:
                q_abs = tl.abs(q_unscaled.to(tl.float32))
                sQ_row = tl.max(q_abs, 1)
                sQ = tl.max(sQ_row) / 127.0
                sQ = tl.where(sQ > 0, sQ, 1e-6)
                q8_tile = libdevice.rint(
                    (q_unscaled.to(tl.float32) / sQ).to(tl.float32)
                ).to(tl.int8)

                k_abs = tl.abs(k.to(tl.float32))
                sK_col = tl.max(k_abs, 0)
                sK = tl.max(sK_col) / 127.0
                sK = tl.where(sK > 0, sK, 1e-6)
                k8_tile = libdevice.rint(
                    (k.to(tl.float32) / sK).to(tl.float32)
                ).to(tl.int8)

                qk_i32 = tl.dot(q8_tile, k8_tile, out_dtype=tl.int32)
                scale = (sQ * sK * qk_scale).to(acc_dtype)
                # Apply the dequantization scale before narrowing to FP16.
                qk += (qk_i32.to(tl.float32) * scale.to(tl.float32)).to(acc_dtype)
            else:
                qk_fp16 = tl.dot(q, k, out_dtype=acc_dtype)
                qk += qk_fp16

        if bias_ptr_iter is not None:
            if boundary_check_needed:
                b = tl.load(
                    bias_ptr_iter, boundary_check=(
                        1,), padding_option="zero")
            else:
                b = tl.load(bias_ptr_iter)
            qk += (b * LOG2E).to(acc_dtype)

        qk_max = tl.max(qk.to(tl.float32), 1)
        m_i_f32 = m_i.to(tl.float32)
        m_ij = tl.maximum(m_i_f32, qk_max).to(acc_dtype)
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk.to(tl.float32)).to(acc_dtype)

        l_ij = tl.sum(p.to(tl.float32), 1).to(acc_dtype)
        if ENABLE_DROPOUT:
            philox_offset = (
                batch_philox_offset
                + start_m * BLOCK_M * seqlen_k
                + start_n
            )
            ms = tl.arange(0, BLOCK_M)
            ns = tl.arange(0, BLOCK_N)
            rng_offsets = (philox_offset +
                           ms[:, None] *
                           seqlen_k +
                           ns[None, :]).to(tl.uint32)
            rng_output = tl.rand(philox_seed, rng_offsets)
            keep = rng_output > dropout_p

            if RETURN_ENCODED_SOFTMAX:
                tl.store(
                    encoded_ptr,
                    tl.where(keep, p, -p).to(encoded_ptr.type.element_ty),
                )
            p = tl.where(keep, p, zero_block)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(
                encoded_ptr,
                p.to(encoded_ptr.type.element_ty),
            )

        alpha = tl.math.exp2((m_i - m_ij).to(tl.float32)).to(acc_dtype)
        acc = acc * alpha[:, None]
        
        if prec_is_int8 and not FUSE_QUANT:
            p_scaled = (p * sv_vec[None, :]).to(acc_dtype)
            v8_fp = v8_tile.to(acc_dtype)
            pv_low = tl.dot(p_scaled, v8_fp).to(acc_dtype)
        elif prec_is_int8 and FUSE_QUANT:
            if PRE_LOAD_V and v_loaded:
                v_fp = v_preload
            else:
                if boundary_check_needed:
                    if PADDED_HEAD:
                        v_fp = tl.load(
                            V_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                    else:
                        v_fp = tl.load(
                            V_ptr,
                            boundary_check=(0,),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                else:
                    if PADDED_HEAD:
                        v_fp = tl.load(
                            V_ptr,
                            boundary_check=(1,),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                    else:
                        v_fp = tl.load(V_ptr, cache_modifier=".ca")
            v_abs = tl.abs(v_fp.to(tl.float32))
            sV_row = tl.max(v_abs, 0)
            sV = tl.max(sV_row) / 127.0
            sV = tl.where(sV > 0, sV, 1e-6)
            v8_fp = libdevice.rint(
                (v_fp.to(tl.float32) / sV).to(tl.float32)
            ).to(tl.int8)
            p_scaled = (p * sV).to(acc_dtype)
            pv_low = tl.dot(p_scaled, v8_fp.to(acc_dtype)).to(acc_dtype)
        else:
            if PRE_LOAD_V and v_loaded:
                v = v_preload
            else:
                if boundary_check_needed:
                    if PADDED_HEAD:
                        v = tl.load(
                            V_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                    else:
                        v = tl.load(
                            V_ptr,
                            boundary_check=(0,),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                else:
                    if PADDED_HEAD:
                        v = tl.load(
                            V_ptr,
                            boundary_check=(1,),
                            padding_option="zero",
                            cache_modifier=".ca",
                        )
                    else:
                        v = tl.load(V_ptr, cache_modifier=".ca")
            pv_low = tl.dot(p.to(v.dtype), v).to(acc_dtype)
        
        l_i = (l_i * alpha + l_ij).to(acc_dtype)
        m_i = m_ij
        acc += pv_low

        V_ptr = tl.advance(V_ptr, (BLOCK_N, 0))
        K_ptr = tl.advance(K_ptr, (0, BLOCK_N))
        if bias_ptr_iter is not None:
            bias_ptr_iter = tl.advance(bias_ptr_iter, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_ptr = tl.advance(encoded_ptr, (0, BLOCK_N))
        K8_ptr = tl.advance(K8_ptr, (0, BLOCK_N))
        V8_ptr = tl.advance(V8_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if (causal_start_idx > start_m_idx) & (causal_start_idx < end_m_idx):
            m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            row_keep = m_offsets >= causal_start_idx
            acc = tl.where(row_keep[:, None], acc, zero_block_dm)

    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    tl.device_assert(start_m * BLOCK_M + BLOCK_M <= seqlen_q + BLOCK_M,
                     "Q-block ptr OOB!")
    tl.store(O_block_ptr, acc, boundary_check=(0, 1))


attn_fwd_with_cache = attn_fwd
attn_fwd_no_cache = attn_fwd


def check_args(
    q,
    k,
    v,
    o,
    varlen=True,
    max_seqlens=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    assert q.dim() == k.dim() and q.dim() == v.dim()
    if varlen:
        assert q.dim() == 3
        total_q, nheads_q, head_size = q.shape
        total_k, nheads_k, _ = k.shape
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
    else:
        assert q.dim() == 4
        batch, nheads_q, seqlen_q, head_size = q.shape
        _, nheads_k, seqlen_k, _ = k.shape
        assert max_seqlens > 0
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]

    assert q.dtype == k.dtype and q.dtype == v.dtype
    assert head_size <= 128
    assert o.shape == q.shape
    assert (nheads_q % nheads_k) == 0


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_n_mask: tl.constexpr,
        causal=False, sm_scale=1.0, bias=None,
        precision_bitmask=None,
        q8=None,
        k8=None,
        v8=None,
        sQ_blocks=None,
        sK_blocks=None,
        sV_blocks=None,
        cu_q_scale=None,
        cu_k_scale=None,
        cu_v_scale=None,
        BLKQ: tl.constexpr = 0,
        BLKK: tl.constexpr = 0,
        candidates=None,
        flat_lengths=None,
    ):

        assert precision_bitmask is not None

        if True:
            if len(cu_seqlens_q) > 1:
                batch_lengths_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                actual_seqlen_q = batch_lengths_q.max().item()
            else:
                actual_seqlen_q = cu_seqlens_q[-1].item() if len(cu_seqlens_q) > 0 else max_seqlens_q
            
            if len(cu_seqlens_k) > 1:
                batch_lengths_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                actual_seqlen_k = batch_lengths_k.max().item()
            else:
                actual_seqlen_k = cu_seqlens_k[-1].item() if len(cu_seqlens_k) > 0 else max_seqlens_k
        else:
            actual_seqlen_q = max_seqlens_q
            actual_seqlen_k = max_seqlens_k

        m_blocks_actual = (actual_seqlen_q + block_m - 1) // block_m
        n_mask_blocks_actual = (actual_seqlen_k + block_n_mask - 1) // block_n_mask

        if actual_seqlen_q != actual_seqlen_k:
            raise ValueError(
                f"seqlen_q ({actual_seqlen_q}) != seqlen_k ({actual_seqlen_k}). "
                f"Bitmask was created assuming they're equal. "
                f"This will cause incorrect indexing."
            )

        if block_n_mask % block_n != 0:
            raise ValueError(f"block_n_mask ({block_n_mask}) must be a multiple of block_n ({block_n})")
        if n_mask_blocks_actual > 64:
            raise ValueError(
                f"64-bit precision mask requires n_mask_blocks<=64, got n_mask_blocks={n_mask_blocks_actual} "
                f"(actual_seqlen_k={actual_seqlen_k}, block_n_mask={block_n_mask})"
            )

        expected_bitmask_elems = k.shape[1] * m_blocks_actual
        assert precision_bitmask.dtype == torch.uint64, "precision_bitmask must be torch.uint64"
        assert precision_bitmask.numel() == expected_bitmask_elems, (
            f"precision_bitmask must be packed as [HK * m_blocks_actual] entries (one uint64 per row), "
            f"got {precision_bitmask.numel()}, expected {expected_bitmask_elems} "
            f"(HK={k.shape[1]}, m_blocks_actual={m_blocks_actual}, n_mask_blocks_actual={n_mask_blocks_actual}, "
            f"actual_seqlen_q={actual_seqlen_q}, actual_seqlen_k={actual_seqlen_k})"
        )
        
        precision_bitmask = (
            precision_bitmask.to(torch.uint64)
            .contiguous()
            .to(q.device)
        )

        cu_seqlens_q = cu_seqlens_q.to(q.device)
        cu_seqlens_k = cu_seqlens_k.to(q.device)

        target_device = q.device
        target_dtype = q.dtype
        
        if q.device != target_device or q.dtype != target_dtype or not q.is_contiguous():
            q = q.to(device=target_device, dtype=target_dtype, non_blocking=True).contiguous()
        if k.device != target_device or k.dtype != target_dtype or not k.is_contiguous():
            k = k.to(device=target_device, dtype=target_dtype, non_blocking=True).contiguous()
        if v.device != target_device or v.dtype != target_dtype or not v.is_contiguous():
            v = v.to(device=target_device, dtype=target_dtype, non_blocking=True).contiguous()
        
        if cu_seqlens_q.device != target_device or not cu_seqlens_q.is_contiguous():
            cu_seqlens_q = cu_seqlens_q.to(device=target_device, dtype=torch.int32, non_blocking=True).contiguous()
        if cu_seqlens_k.device != target_device or not cu_seqlens_k.is_contiguous():
            cu_seqlens_k = cu_seqlens_k.to(device=target_device, dtype=torch.int32, non_blocking=True).contiguous()
        if precision_bitmask.device != target_device or not precision_bitmask.is_contiguous():
            precision_bitmask = precision_bitmask.to(device=target_device, dtype=torch.uint64, non_blocking=True).contiguous()

        assert all(t is not None for t in (q8, k8, v8)), "Missing int8 tensors"
        assert all(t is not None for t in (sQ_blocks, sK_blocks, sV_blocks)), \
            "Provide compact per-block scales for INT8 path"
        assert all(t is not None for t in (cu_q_scale, cu_k_scale, cu_v_scale)), \
            "Provide per-batch scale prefix sums"
        assert BLKQ > 0 and BLKK > 0, "Quantization block sizes must be positive"

        target_device = q.device
        if q8.device != target_device or not q8.is_contiguous():
            q8 = q8.to(device=target_device, dtype=torch.int8, non_blocking=True).contiguous()
        if k8.device != target_device or not k8.is_contiguous():
            k8 = k8.to(device=target_device, dtype=torch.int8, non_blocking=True).contiguous()
        if v8.device != target_device or not v8.is_contiguous():
            v8 = v8.to(device=target_device, dtype=torch.int8, non_blocking=True).contiguous()
        
        if sQ_blocks.device != target_device or not sQ_blocks.is_contiguous():
            sQ_blocks = sQ_blocks.to(device=target_device, non_blocking=True).contiguous()
        if sK_blocks.device != target_device or not sK_blocks.is_contiguous():
            sK_blocks = sK_blocks.to(device=target_device, non_blocking=True).contiguous()
        if sV_blocks.device != target_device or not sV_blocks.is_contiguous():
            sV_blocks = sV_blocks.to(device=target_device, non_blocking=True).contiguous()
        
        if cu_q_scale.device != target_device or not cu_q_scale.is_contiguous():
            cu_q_scale = cu_q_scale.to(device=target_device, dtype=torch.int32, non_blocking=True).contiguous()
        if cu_k_scale.device != target_device or not cu_k_scale.is_contiguous():
            cu_k_scale = cu_k_scale.to(device=target_device, dtype=torch.int32, non_blocking=True).contiguous()
        if cu_v_scale.device != target_device or not cu_v_scale.is_contiguous():
            cu_v_scale = cu_v_scale.to(device=target_device, dtype=torch.int32, non_blocking=True).contiguous()
        assert q8.dtype == torch.int8 and k8.dtype == torch.int8
        assert v8.dtype == torch.int8
        assert sQ_blocks.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert sK_blocks.dtype in (torch.float16, torch.bfloat16, torch.float32)
        assert sV_blocks.dtype in (torch.float16, torch.bfloat16, torch.float32)
        
        if True:
            total_q, nheads_q, head_size = q.shape
            total_k, nheads_k, _ = k.shape
            batch = len(cu_seqlens_q) - 1
            
            assert sQ_blocks.shape[1] == nheads_q, \
                f"sQ_blocks head dim mismatch: got {sQ_blocks.shape[1]}, expected {nheads_q}"
            assert sK_blocks.shape[1] == nheads_k, \
                f"sK_blocks head dim mismatch: got {sK_blocks.shape[1]}, expected {nheads_k}"
            assert sV_blocks.shape[1] == nheads_k, \
                f"sV_blocks head dim mismatch: got {sV_blocks.shape[1]}, expected {nheads_k}"
            
            max_q_blocks = cu_q_scale[-1].item() if cu_q_scale.numel() > 0 else 0
            max_k_blocks = cu_k_scale[-1].item() if cu_k_scale.numel() > 0 else 0
            max_v_blocks = cu_v_scale[-1].item() if cu_v_scale.numel() > 0 else 0

            if (max_q_blocks <= 0) or (max_q_blocks > cu_seqlens_q[-1].item() // BLKQ + 4):
                max_q_blocks = (cu_seqlens_q[-1].item() + BLKQ - 1) // BLKQ
            if (max_k_blocks <= 0) or (max_k_blocks > cu_seqlens_k[-1].item() // BLKK + 4):
                max_k_blocks = (cu_seqlens_k[-1].item() + BLKK - 1) // BLKK
            if (max_v_blocks <= 0) or (max_v_blocks > cu_seqlens_k[-1].item() // BLKK + 4):
                max_v_blocks = (cu_seqlens_k[-1].item() + BLKK - 1) // BLKK
            
            assert sQ_blocks.shape[0] >= max_q_blocks, \
                f"sQ_blocks block dim too small: got {sQ_blocks.shape[0]}, need at least {max_q_blocks}"
            assert sK_blocks.shape[0] >= max_k_blocks, \
                f"sK_blocks block dim too small: got {sK_blocks.shape[0]}, need at least {max_k_blocks}"
            assert sV_blocks.shape[0] >= max_v_blocks, \
                f"sV_blocks block dim too small: got {sV_blocks.shape[0]}, need at least {max_v_blocks}"

        if o is None:
            o = torch.empty_like(q, dtype=v.dtype)

        check_args(
            q,
            k,
            v,
            o,
            varlen=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
        if True:
            total_q, nheads_q, head_size = q.shape
            assert head_size <= 128 and head_size > 0
            total_k, nheads_k, _ = k.shape
            batch = len(cu_seqlens_q) - 1
            n_blocks_per_k = (max_seqlens_k + block_n - 1) // block_n
            q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
            k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
            v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
            o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
            q8_strides = (0, q8.stride(1), q8.stride(0), q8.stride(2))
            k8_strides = (0, k8.stride(1), k8.stride(0), k8.stride(2))
            v8_strides = (0, v8.stride(1), v8.stride(0), v8.stride(2))
        else:
            batch, seqlen_q, nheads_q, head_size = q.shape
            _, seqlen_k, nheads_k, _ = k.shape
            n_blocks_per_k = (max_seqlens_k + block_n - 1) // block_n
            q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
            k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
            v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
            o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
            q8_strides = (
                q8.stride(0),
                q8.stride(2),
                q8.stride(1),
                q8.stride(3))
            k8_strides = (
                k8.stride(0),
                k8.stride(2),
                k8.stride(1),
                k8.stride(3))
            v8_strides = (
                v8.stride(0),
                v8.stride(2),
                v8.stride(1),
                v8.stride(3))

        padded_d_model = 1 << (head_size - 1).bit_length()
        padded_d_model = max(padded_d_model, 32)
        padded_d_model = ((padded_d_model + 31) // 32) * 32
        assert (
            padded_d_model %
            32) == 0, "INT8 IMMA requires K multiple of 32"

        def grid(META):
            g = (
                triton.cdiv(max_seqlens_q, META["BLOCK_M"]),
                nheads_q,
                batch,
            )
            return g

        encoded_softmax = None

        philox_seed = 0x1BF52
        philox_offset = 0x1D4B42

        if bias is not None:
            bias_strides = (
                bias.stride(0),
                bias.stride(1),
                bias.stride(2),
                bias.stride(3),
            )
        else:
            bias_strides = (0, 0, 0, 0)

        n_blocks_per_k = (max_seqlens_k + block_n - 1) // block_n
        kernel = attn_fwd

        compute_capability = None
        if q.is_cuda:
            device_index = q.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device_index)
        cc_major = compute_capability[0] if compute_capability is not None else 0
        warp_specialize = 1 if (cc_major >= 9 and causal) else 0

        forced = os.environ.get("TILEMIX_FORCE_CONFIG")
        if forced:
            try:
                parts = [p.strip() for p in forced.split(",")]
                bm = int(parts[0]); bn = int(parts[1])
                warps = int(parts[2]) if len(parts) > 2 and parts[2] != "" else 8
                stages = int(parts[3]) if len(parts) > 3 and parts[3] != "" else 2
                preload_raw = parts[4].lower() if len(parts) > 4 else "0"
                preload = preload_raw in ("1", "true", "yes", "y", "on")
                kernel.configs = [
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "PRE_LOAD_V": preload},
                        num_stages=stages, num_warps=warps
                    )
                ]
            except Exception:
                kernel.configs = _get_filtered_configs(
                    max_seqlens_q=max_seqlens_q,
                    max_seqlens_k=max_seqlens_k,
                    compute_capability=compute_capability,
                )
        else:
            desired_bm = int(block_m)
            desired_bn = int(block_n)
            matching = [
                cfg
                for cfg in ALL_AUTOTUNE_CONFIGS
                if cfg.kwargs.get("BLOCK_M") == desired_bm
                and cfg.kwargs.get("BLOCK_N") == desired_bn
            ]
            if matching:
                def _legal(cfg: triton.Config) -> bool:
                    kwargs = cfg.kwargs
                    if kwargs["BLOCK_M"] > max_seqlens_q:
                        return False
                    if kwargs["BLOCK_N"] > max_seqlens_k:
                        return False
                    if compute_capability is not None:
                        cc_major = compute_capability[0]
                        if cc_major < 8 and kwargs.get("PRE_LOAD_V", False):
                            return False
                        if cc_major < 9 and getattr(cfg, "num_warps", 8) > 8:
                            return False
                    return True
                legal_matching = [c for c in matching if _legal(c)]
                if legal_matching:
                    kernel.configs = legal_matching
                else:
                    kernel.configs = [
                        triton.Config(
                            {"BLOCK_M": desired_bm, "BLOCK_N": desired_bn, "PRE_LOAD_V": False},
                            num_stages=2,
                            num_warps=8,
                        )
                    ]
            else:
                kernel.configs = [
                    triton.Config(
                        {"BLOCK_M": desired_bm, "BLOCK_N": desired_bn, "PRE_LOAD_V": False},
                        num_stages=2,
                        num_warps=8,
                    )
                ]

        fuse_quant_env = os.environ.get("TILEMIX_FUSE_QUANT", "0").strip().lower()
        fuse_quant = 0 if fuse_quant_env in ("0", "false", "no", "") else 1

        host_grid = (
            triton.cdiv(max_seqlens_q, block_m),
            nheads_q,
            batch,
        )

        try:
            kernel[grid](
                q,
                k,
                v,
                bias,
                sm_scale,
                None,
                o,
                *q_strides,
                *k_strides,
                *v_strides,
                *o_strides,
                *bias_strides,
                cu_seqlens_q,
                cu_seqlens_k,
                dropout_p=0.0,
                philox_seed=philox_seed,
                philox_offset_base=philox_offset,
                encoded_softmax=None,
                precision_bitmask=precision_bitmask,
                Q8=q8,
                K8=k8,
                V8=v8,
                sQ_blocks=sQ_blocks,
                sK_blocks=sK_blocks,
                sV_blocks=sV_blocks,
                cu_q_scale=cu_q_scale,
                cu_k_scale=cu_k_scale,
                cu_v_scale=cu_v_scale,
                stride_q8z=q8_strides[0], stride_q8h=q8_strides[1], stride_q8m=q8_strides[2], stride_q8k=q8_strides[3],
                stride_k8z=k8_strides[0], stride_k8h=k8_strides[1], stride_k8n=k8_strides[2], stride_k8k=k8_strides[3],
                stride_v8z=v8_strides[0], stride_v8h=v8_strides[1], stride_v8k=v8_strides[2], stride_v8n=v8_strides[3],
                HQ=nheads_q,
                HK=nheads_k,
                ACTUAL_BLOCK_DMODEL=head_size,
                MAX_SEQLENS_Q=max_seqlens_q,
                MAX_SEQLENS_K=max_seqlens_k,
                IS_CAUSAL=causal,
                VARLEN=True,
                BLOCK_DMODEL=padded_d_model,
                BLKQ=BLKQ,
                BLKK=BLKK,
                WARP_SPECIALIZE=warp_specialize,
                FUSE_QUANT=fuse_quant,
                EVEN_M=int((max_seqlens_q % block_m) == 0),
                EVEN_N=int((max_seqlens_k % block_n) == 0),
                EVEN_DMODEL=int(head_size == padded_d_model),
                BLOCK_N_MASK=block_n_mask,
                BIAS_TYPE=0 if bias is None else 1,
                ENABLE_DROPOUT=False,
                RETURN_ENCODED_SOFTMAX=False,
            )
        except Exception as err:
            message = str(err)
            err_type = type(err).__name__
            is_out_of_resources = (
                isinstance(err, OutOfResources) or 
                err_type == "OutOfResources" or
                "OutOfResources" in message or 
                "out of resource" in message.lower()
            )
            raise

        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = head_size
        ctx.causal = causal
        ctx.dropout_p = 0.0
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax
        ctx.return_encoded_softmax = False
        return o, None
@torch.no_grad()
def quantize_qkv_for_tilemix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale: float = None,
):
    """
    Quantize q, k, and v per block and return compact scale metadata.

    Returns:
        Tuple containing int8 tensors, per-block scales, and block prefix sums:
        (q8, q_scale_blocks, cu_q_scale,
         k8, k_scale_blocks, cu_k_scale,
         v8, v_scale_blocks, cu_v_scale)
    """

    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    q8, q_scale_blocks, k8, k_scale_blocks, cu_q_scale, cu_k_scale = quant_per_block_int8(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLKQ=BLKQ,
        BLKK=BLKK,
        sm_scale=1.0,
    )

    v8, v_scale_blocks, cu_v_scale = per_block_int8_single(
        v,
        cu_seqlens_k,
        max_seqlen_k,
        BLK=BLKK,
        sm_scale=1.0,
    )

    if q.dtype == torch.float16:
        scale_dtype = torch.float16
    elif q.dtype == torch.bfloat16:
        scale_dtype = torch.bfloat16
    else:
        scale_dtype = torch.float32

    q_scale_blocks = q_scale_blocks.to(dtype=scale_dtype).contiguous()
    k_scale_blocks = k_scale_blocks.to(dtype=scale_dtype).contiguous()
    v_scale_blocks = v_scale_blocks.to(dtype=scale_dtype).contiguous()

    cu_q_scale = cu_q_scale.to(dtype=torch.int32).contiguous()
    cu_k_scale = cu_k_scale.to(dtype=torch.int32).contiguous()
    cu_v_scale = cu_v_scale.to(dtype=torch.int32).contiguous()

    return (
        q8,
        q_scale_blocks,
        cu_q_scale,
        k8,
        k_scale_blocks,
        cu_k_scale,
        v8,
        v_scale_blocks,
        cu_v_scale,
    )


@torch.no_grad()
def quantize_qk_for_tilemix(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale: float = None,
):
    """Quantize only the routed score operands, leaving PV in FP16."""

    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    q8, q_scale_blocks, k8, k_scale_blocks, cu_q_scale, cu_k_scale = (
        quant_per_block_int8(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            BLKQ=BLKQ,
            BLKK=BLKK,
            # Match quantize_qkv_for_tilemix: score scaling is applied after
            # the INT32 QK accumulator inside the attention kernel.
            sm_scale=1.0,
        )
    )

    if q.dtype == torch.float16:
        scale_dtype = torch.float16
    elif q.dtype == torch.bfloat16:
        scale_dtype = torch.bfloat16
    else:
        scale_dtype = torch.float32

    return (
        q8.contiguous(),
        q_scale_blocks.to(dtype=scale_dtype).contiguous(),
        cu_q_scale.to(dtype=torch.int32).contiguous(),
        k8.contiguous(),
        k_scale_blocks.to(dtype=scale_dtype).contiguous(),
        cu_k_scale.to(dtype=torch.int32).contiguous(),
    )


@torch.no_grad()
def quantize_q_for_tilemix(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    BLKQ: int = 128,
    sm_scale: float = None,
):
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    q8, q_scale_blocks, cu_q_scale = per_block_int8_single(
        q,
        cu_seqlens_q,
        max_seqlen_q,
        BLK=BLKQ,
        sm_scale=(sm_scale * 1.0),
    )
    if q.dtype == torch.float16:
        scale_dtype = torch.float16
    elif q.dtype == torch.bfloat16:
        scale_dtype = torch.bfloat16
    else:
        scale_dtype = torch.float32
    return (
        q8.contiguous(),
        q_scale_blocks.to(dtype=scale_dtype).contiguous(),
        cu_q_scale.to(dtype=torch.int32).contiguous(),
    )


@torch.no_grad()
def tilemix_attention_with_cached_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlens_q: int,
    max_seqlens_k: int,
    precision_bitmask: torch.Tensor,
    *,
    q_blk: int = 128,
    k_blk: int = 64,
    block_m: int = 128,
    block_n: int = 512,
    block_n_mask: int | None = None,
    sm_scale: float = None,
    causal: bool = True,
    bias: torch.Tensor | None = None,
    k8: torch.Tensor,
    v8: torch.Tensor,
    sQ_blocks: torch.Tensor,
    sK_blocks: torch.Tensor,
    sV_blocks: torch.Tensor,
    cu_q_scale: torch.Tensor,
    cu_k_scale: torch.Tensor,
    cu_v_scale: torch.Tensor,
):
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    if block_n_mask is None:
        block_n_mask = block_n

    q8, q_scale_blocks, cu_q_scale_new = quantize_q_for_tilemix(
        q, cu_seqlens_q, max_seqlens_q, BLKQ=q_blk, sm_scale=sm_scale
    )

    if q.dtype == torch.float16:
        scale_dtype = torch.float16
    elif q.dtype == torch.bfloat16:
        scale_dtype = torch.bfloat16
    else:
        scale_dtype = torch.float32

    q_scale_blocks = q_scale_blocks.to(dtype=scale_dtype).contiguous()
    sK_blocks = sK_blocks.to(dtype=scale_dtype).contiguous()
    sV_blocks = sV_blocks.to(dtype=scale_dtype).contiguous()

    cu_q_scale_new = cu_q_scale_new.to(dtype=torch.int32).contiguous()
    cu_k_scale = cu_k_scale.to(dtype=torch.int32).contiguous()
    cu_v_scale = cu_v_scale.to(dtype=torch.int32).contiguous()

    out, _ = triton_attention(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        block_m,
        block_n,
        block_n_mask,
        causal,
        sm_scale,
        bias,
        precision_bitmask,
        q8,
        k8,
        v8,
        q_scale_blocks,
        sK_blocks,
        sV_blocks,
        cu_q_scale_new,
        cu_k_scale,
        cu_v_scale,
        q_blk,
        k_blk,
        None,
        None,
    )
    return out


def tilemix_attention_with_quant(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlens_q,
    max_seqlens_k,
    precision_bitmask,
    block_m=128,
    block_n=128,
    block_n_mask=None,
    BLKQ=128,
    BLKK=64,
    causal=False,
    sm_scale=None,
    bias=None,
    o=None,
    fuse_quant: bool = False,
):
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    if block_n_mask is None:
        block_n_mask = block_n

    if not fuse_quant:
        (
            q8,
            q_scale_blocks,
            cu_q_scale,
            k8,
            k_scale_blocks,
            cu_k_scale,
            v8,
            v_scale_blocks,
            cu_v_scale,
        ) = quantize_qkv_for_tilemix(
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
    else:
        # The common launcher still constructs block pointers and validates
        # tensor ranks before the FUSE_QUANT constexpr removes all external
        # quantized-operand loads.  Use minimal correctly ranked placeholders
        # instead of rank-1 empty tensors; no full Q8/K8/V8 copies are needed.
        q8 = torch.empty(
            (1, q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.int8,
        )
        k8 = torch.empty(
            (1, k.shape[1], k.shape[2]),
            device=k.device,
            dtype=torch.int8,
        )
        v8 = torch.empty(
            (1, v.shape[1], v.shape[2]),
            device=v.device,
            dtype=torch.int8,
        )
        q_block_rows = (int(cu_seqlens_q[-1].item()) + BLKQ - 1) // BLKQ
        k_block_rows = (int(cu_seqlens_k[-1].item()) + BLKK - 1) // BLKK
        q_scale_blocks = torch.empty(
            (q_block_rows, q.shape[1]),
            device=q.device,
            dtype=q.dtype,
        )
        k_scale_blocks = torch.empty(
            (k_block_rows, k.shape[1]),
            device=k.device,
            dtype=k.dtype,
        )
        v_scale_blocks = torch.empty_like(k_scale_blocks)
        # These prefixes are loaded by common pointer setup but are not used
        # by the fused specialization.
        cu_q_scale = torch.zeros_like(cu_seqlens_q, dtype=torch.int32)
        cu_k_scale = torch.zeros_like(cu_seqlens_k, dtype=torch.int32)
        cu_v_scale = torch.zeros_like(cu_seqlens_k, dtype=torch.int32)

    previous_fuse_quant = os.environ.get("TILEMIX_FUSE_QUANT")
    os.environ["TILEMIX_FUSE_QUANT"] = "1" if fuse_quant else "0"
    try:
        out, _ = triton_attention(
            q,
            k,
            v,
            o,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlens_q,
            max_seqlens_k,
            block_m,
            block_n,
            block_n_mask,
            causal,
            sm_scale,
            bias,
            precision_bitmask,
            q8,
            k8,
            v8,
            q_scale_blocks,
            k_scale_blocks,
            v_scale_blocks,
            cu_q_scale,
            cu_k_scale,
            cu_v_scale,
            BLKQ,
            BLKK,
            None,
            None,
        )
    finally:
        if previous_fuse_quant is None:
            os.environ.pop("TILEMIX_FUSE_QUANT", None)
        else:
            os.environ["TILEMIX_FUSE_QUANT"] = previous_fuse_quant

    return out


triton_attention = _attention.apply
