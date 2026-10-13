import torch
import triton
import triton.language as tl


@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale,
                                cu_seqlens_input, cu_seqlens_scale,
                                stride_ih, stride_in,
                                stride_oh, stride_on,
                                sm_scale,
                                H: tl.constexpr,
                                C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    cu_seqlens_input_start = tl.load(cu_seqlens_input + off_b)
    cu_seqlens_input_end = tl.load(cu_seqlens_input + off_b + 1)

    L = cu_seqlens_input_end - cu_seqlens_input_start

    if (off_blk * BLK) >= L:
        return

    cu_seqlens_scale_start = tl.load(cu_seqlens_scale + off_b)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + cu_seqlens_input_start * stride_in + off_h * \
        stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + cu_seqlens_input_start * stride_on + \
        off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + cu_seqlens_scale_start * H + off_h + off_blk * H

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def per_block_int8_single(
        x, cu_seqlens_x, max_seqlen_x, BLK=128, sm_scale=1.0):
    x_int8 = torch.empty(x.shape, dtype=torch.int8, device=x.device)

    h = x.shape[1]
    head_dim = x.shape[-1]

    b = cu_seqlens_x.shape[0] - 1
    x_batch_len = cu_seqlens_x[1:] - cu_seqlens_x[:-1]
    x_scale_len = (x_batch_len + BLK - 1) // BLK

    cu_seqlens_x_scale = torch.nn.functional.pad(
        torch.cumsum(x_scale_len, dim=0), (1, 0), value=0)

    x_scale = torch.empty(
        (cu_seqlens_x_scale[-1], h), device=x.device, dtype=torch.float32)

    grid = ((max_seqlen_x + BLK - 1) // BLK, h, b)
    quant_per_block_int8_kernel[grid](
        x, x_int8, x_scale,
        cu_seqlens_x, cu_seqlens_x_scale,
        x.stride(1), x.stride(0),
        x_int8.stride(1), x_int8.stride(0),
        sm_scale=sm_scale, H=h,
        C=head_dim, BLK=BLK
    )

    return x_int8, x_scale, cu_seqlens_x_scale


def per_block_int8(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                   max_seqlen_k, BLKQ=128, BLKK=64, sm_scale=None):
    head_dim = q.shape[-1]

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if abs(sm_scale - 1.0) < 1e-6:
        q_sm_scale = 1.0
    else:
        q_sm_scale = sm_scale * 1.44269504

    q_int8, q_scale, cu_seqlens_q_scale = per_block_int8_single(
        q,
        cu_seqlens_q,
        max_seqlen_q,
        BLK=BLKQ,
        sm_scale=q_sm_scale,
    )

    k_int8, k_scale, cu_seqlens_k_scale = per_block_int8_single(
        k,
        cu_seqlens_k,
        max_seqlen_k,
        BLK=BLKK,
        sm_scale=1.0,
    )

    return q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale
