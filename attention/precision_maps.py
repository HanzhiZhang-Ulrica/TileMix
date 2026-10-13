import torch
import random


def gen_one_pattern(num_query, num_key,
                    dtype: torch.dtype = torch.bool,
                    num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    return torch.ones((num_layer, num_head, num_query, num_key), dtype=dtype)


def gen_zero_pattern(num_query, num_key,
                     dtype: torch.dtype = torch.bool,
                     num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    return torch.zeros((num_layer, num_head, num_query, num_key), dtype=dtype)


def gen_causal_pattern(num_query, num_key,
                       dtype: torch.dtype = torch.bool,
                       num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    q_idx = torch.arange(num_query).unsqueeze(1)
    k_idx = torch.arange(num_key).unsqueeze(0)
    mask2d = (k_idx > q_idx)
    return mask2d[None, None]                     \
        .expand(num_layer, num_head, -1, -1) \
        .to(dtype)


def calculate_band_width_from_percentage(
    seq_len: int,
    percentage: float,
    block_n: int
) -> int:
    if percentage is None or percentage <= 0.0 or percentage >= 1.0:
        return 5

    if block_n is None or block_n <= 0:
        target_band_ratio = 1.0 - percentage
        estimated_band_width = int(target_band_ratio * seq_len)
        if estimated_band_width % 2 == 0:
            estimated_band_width += 1
        return max(3, min(estimated_band_width, seq_len))

    num_n = (seq_len + block_n - 1) // block_n
    target_band_ratio = 1.0 - percentage
    target_band_blocks = target_band_ratio * num_n

    band_blocks = int(round(target_band_blocks))
    if band_blocks % 2 == 0:
        if target_band_blocks > band_blocks:
            band_blocks += 1
        else:
            band_blocks = max(1, band_blocks - 1)

    band_blocks = max(1, min(band_blocks, num_n))
    band_width = band_blocks * block_n

    if band_width % 2 == 0:
        band_width += 1

    band_width = max(3, min(band_width, seq_len))

    return band_width


def gen_band_pattern(num_query, num_key, band_width: int = None,
                     dtype: torch.dtype = torch.bool,
                     num_layer: int = 1, num_head: int = 1,
                     percentage: float = None,
                     block_n: int = None) -> torch.Tensor:
    if band_width is None and percentage is not None:
        band_width = calculate_band_width_from_percentage(num_query, percentage, block_n)

    if band_width is None:
        band_width = 5

    if band_width % 2 == 0:
        band_width += 1
    band_width = min(band_width, num_key)

    q = torch.arange(num_query)
    k = torch.arange(num_key)
    start = q[:, None] - band_width // 2
    end = q[:, None] + band_width // 2 + 1
    mask2d = (k[None, :] >= start) & (k[None, :] < end)
    return mask2d[None, None].expand(num_layer, num_head, -1, -1).to(dtype)


def gen_global_pattern(num_query, num_key, key_pos: list,
                       dtype: torch.dtype = torch.bool,
                       num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    valid_key_pos = [pos for pos in key_pos if pos < num_key]

    g = torch.zeros(
        (num_layer,
         num_head,
         num_query,
         num_key),
        dtype=torch.bool)

    if valid_key_pos:
        g[:, :, :, valid_key_pos] = True

    return g.to(dtype)


def gen_row_rand_pattern(num_query, num_key, num_rand_block_per_row: int,
                         dtype: torch.dtype = torch.bool,
                         num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    rand = torch.zeros(
        (num_layer,
         num_head,
         num_query,
         num_key),
        dtype=torch.bool)
    for l in range(num_layer):
        for h in range(num_head):
            for i in range(num_query):
                idx = random.sample(range(num_key), num_rand_block_per_row)
                rand[l, h, i, idx] = True
    return rand.to(dtype)


def gen_align_sparse_pattern(num_query, num_key, alpha: int, beta: float,
                             dtype: torch.dtype = torch.bool,
                             num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    pat = torch.zeros(
        (num_layer,
         num_head,
         num_query,
         num_key),
        dtype=torch.bool)
    for L in range(num_layer):
        for H in range(num_head):
            for i in range(num_query):
                eff = min(i + 1, num_key)
                nnz = int(min(alpha + beta * (eff - alpha) * (eff > alpha), eff))
                start = max(0, min(i - nnz + 1, num_key - nnz))
                pat[L, H, i, start:start + nnz] = True
    return pat.to(dtype)


def gen_bigbird_pattern(num_query, num_key,
                        num_band_block: int = 3,
                        num_global_block: int = 2,
                        num_rand_block: int = 3,
                        dtype: torch.dtype = torch.bool,
                        num_layer: int = 1, num_head: int = 1) -> torch.Tensor:
    band = gen_band_pattern(
        num_query,
        num_key,
        num_band_block,
        dtype,
        num_layer,
        num_head)

    actual_global_block = min(num_global_block, num_key)
    glob = gen_global_pattern(
        num_query,
        num_key,
        list(
            range(actual_global_block)),
        dtype,
        num_layer,
        num_head)
    glob = glob | glob.transpose(-2, -1)

    actual_rand_block = min(num_rand_block, num_key)
    rand = gen_row_rand_pattern(
        num_query,
        num_key,
        actual_rand_block,
        dtype,
        num_layer,
        num_head)

    return (band | glob | rand).to(dtype)


def gen_block_pattern(layout: torch.Tensor, block_size: int = 64,
                      dtype: torch.dtype = torch.bool) -> torch.Tensor:
    if layout.ndim == 2:
        layout = layout.unsqueeze(0).unsqueeze(0)
    if layout.ndim == 3:
        layout = layout.unsqueeze(0)
    L, H, QB, KB = layout.shape
    m = layout.unsqueeze(-1).unsqueeze(-1) \
        .permute(0, 1, 2, 4, 3, 5) \
        .repeat(1, 1, 1, block_size, 1, block_size) \
        .view(L, H, QB * block_size, KB * block_size)
    return m.to(dtype)


def layout_to_causal_mask(layout: torch.Tensor, block_size: int,
                          dtype: torch.dtype = torch.bool) -> torch.Tensor:
    L, H, QB, KB = layout.shape
    full = gen_block_pattern(layout, block_size, dtype)
    causal = gen_causal_pattern(QB * block_size, KB * block_size, dtype, L, H)
    return full & causal


def _apply_percentage_control(
    block_precision_map: torch.Tensor,
    percentage: float,
) -> torch.Tensor:
    if percentage is None or percentage < 0.0 or percentage > 1.0:
        return block_precision_map

    num_m, num_n = block_precision_map.shape
    total_blocks = num_m * num_n

    current_low_prec = block_precision_map.sum().item()
    target_low_prec = int(total_blocks * percentage)

    if current_low_prec >= target_low_prec:
        return block_precision_map

    num_to_flip = target_low_prec - current_low_prec

    high_prec_mask = ~block_precision_map
    high_prec_indices = torch.nonzero(high_prec_mask, as_tuple=False)

    if len(high_prec_indices) == 0:
        return block_precision_map

    if num_to_flip > len(high_prec_indices):
        num_to_flip = len(high_prec_indices)

    perm = torch.randperm(len(high_prec_indices), device=block_precision_map.device)
    selected_indices = high_prec_indices[perm[:num_to_flip]]

    result = block_precision_map.clone()
    for idx in selected_indices:
        result[idx[0], idx[1]] = True

    return result


def _causal_compute_tile_weights(
    seq_len: int,
    block_m: int,
    block_n_mask: int,
    compute_block_n: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if block_n_mask % compute_block_n != 0:
        raise ValueError(
            f"block_n_mask ({block_n_mask}) must be a multiple of "
            f"compute_block_n ({compute_block_n})"
        )
    num_m = (seq_len + block_m - 1) // block_m
    num_mask_n = (seq_len + block_n_mask - 1) // block_n_mask
    num_compute_n = (seq_len + compute_block_n - 1) // compute_block_n
    mask_group = block_n_mask // compute_block_n
    weights = torch.zeros(
        (num_m, num_mask_n),
        dtype=torch.int64,
        device=device,
    )
    for row in range(num_m):
        row_compute_tiles = min(
            num_compute_n,
            ((row + 1) * block_m + compute_block_n - 1) // compute_block_n,
        )
        full_groups, remainder = divmod(row_compute_tiles, mask_group)
        if full_groups:
            weights[row, :full_groups] = mask_group
        if remainder and full_groups < num_mask_n:
            weights[row, full_groups] = remainder
    return weights


def _retarget_weighted_int8_fraction(
    block_precision_map: torch.Tensor,
    weights: torch.Tensor,
    percentage: float,
    *,
    seed: int = 42,
) -> torch.Tensor:
    if percentage < 0.0 or percentage > 1.0:
        raise ValueError(f"percentage must be in [0, 1], got {percentage}")
    if block_precision_map.shape != weights.shape:
        raise ValueError(
            "block_precision_map and weights must have identical shapes, got "
            f"{tuple(block_precision_map.shape)} and {tuple(weights.shape)}"
        )

    original_device = block_precision_map.device
    result = block_precision_map.detach().to(device="cpu", dtype=torch.bool).clone()
    host_weights = weights.detach().to(device="cpu", dtype=torch.int64)
    visited = host_weights > 0
    result &= visited
    total_weight = int(host_weights.sum().item())
    if total_weight == 0:
        return result.to(device=original_device)

    target = int(round(total_weight * percentage))
    current = int((host_weights * result).sum().item())
    if current == target:
        return result.to(device=original_device)

    add_int8 = current < target
    candidates = torch.nonzero(
        ((~result) if add_int8 else result) & visited,
        as_tuple=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(candidates), generator=generator)
    for position in order.tolist():
        row, col = candidates[position].tolist()
        weight = int(host_weights[row, col].item())
        proposed = current + weight if add_int8 else current - weight
        if abs(target - proposed) <= abs(target - current):
            result[row, col] = add_int8
            current = proposed
        if current == target:
            break

    max_weight = int(host_weights.max().item())
    if abs(current - target) > max_weight:
        raise RuntimeError(
            "unable to retarget executable INT8 fraction within one mask-bit "
            f"weight: target={target}, achieved={current}, max_weight={max_weight}"
        )
    return result.to(device=original_device)


def _make_sptrans_block_precision_map(
    seq_len: int,
    block_m: int,
    block_n_mask: int,
    compute_block_n: int,
    percentage: float,
    pattern_params: dict,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    num_m = (seq_len + block_m - 1) // block_m
    num_n = (seq_len + block_n_mask - 1) // block_n_mask
    sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=device)
    stride = pattern_params.get("stride", min(128, seq_len // 4))
    c = pattern_params.get("c", min(32, stride // 4))

    for row in range(num_m):
        query_start = row * block_m
        query_end = min(seq_len, (row + 1) * block_m)
        first_stride = query_start // stride
        last_stride = (query_end - 1) // stride
        for query_stride in range(first_stride, last_stride + 1):
            stride_start = query_stride * stride
            stride_end = min((query_stride + 1) * stride, query_end)
            start_block = stride_start // block_n_mask
            end_block = min(
                num_n,
                (stride_end + block_n_mask - 1) // block_n_mask,
            )
            if start_block < num_n:
                sparse[row, start_block:end_block] = False

            for previous_stride in range(query_stride):
                previous_start = previous_stride * stride
                previous_end = (previous_stride + 1) * stride
                tail_start = max(previous_start, previous_end - c)
                tail_start_block = tail_start // block_n_mask
                tail_end_block = min(
                    num_n,
                    (previous_end + block_n_mask - 1) // block_n_mask,
                )
                if tail_start_block < num_n:
                    sparse[row, tail_start_block:tail_end_block] = False

    weights = _causal_compute_tile_weights(
        seq_len,
        block_m,
        block_n_mask,
        compute_block_n,
        device=device,
    )
    return _retarget_weighted_int8_fraction(
        sparse,
        weights,
        percentage,
        seed=42,
    )


def pack_block_precision_map(block_precision_map: torch.Tensor) -> torch.Tensor:
    if block_precision_map.ndim != 2 or block_precision_map.shape[1] > 64:
        raise ValueError(
            "block_precision_map must be 2D with at most 64 columns, got "
            f"{tuple(block_precision_map.shape)}"
        )
    original_device = block_precision_map.device
    packed = []
    for row in block_precision_map.detach().to(device="cpu", dtype=torch.bool).tolist():
        value = 0
        for bit, route in enumerate(row):
            value |= int(route) << bit
        packed.append(value)
    return torch.tensor(packed, dtype=torch.uint64, device=original_device)


def make_precision_bitmask(
    seq_len: int,
    BLOCK_M: int,
    BLOCK_N: int,
    token_mask: torch.Tensor | None,
    percentage: float = None,
    pattern_type: str = None,
    pattern_params: dict = None,
    COMPUTE_BLOCK_N: int = None,
) -> torch.Tensor:
    num_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    if COMPUTE_BLOCK_N is None:
        COMPUTE_BLOCK_N = BLOCK_N
    if BLOCK_N % COMPUTE_BLOCK_N != 0:
        raise ValueError(
            f"BLOCK_N(mask) ({BLOCK_N}) must be a multiple of "
            f"COMPUTE_BLOCK_N ({COMPUTE_BLOCK_N})"
        )
    if num_n > 64:
        raise ValueError(
            f"num_n_mask={num_n} > 64; BLOCK_N(mask) too small "
            f"(seq_len={seq_len}, BLOCK_N(mask)={BLOCK_N})"
        )
    if pattern_type in {"zero", "one"}:
        weights = _causal_compute_tile_weights(
            seq_len,
            BLOCK_M,
            BLOCK_N,
            COMPUTE_BLOCK_N,
        )
        sparse = (
            torch.zeros_like(weights, dtype=torch.bool)
            if pattern_type == "zero"
            else weights > 0
        )
        return pack_block_precision_map(sparse)

    block_percentage_patterns = {
        "band",
        "global",
        "row_rand",
        "align_sparse",
        "bigbird",
        "sptrans",
    }
    if (
        pattern_type == "sptrans"
        and percentage is not None
        and pattern_params is not None
    ):
        target_device = token_mask.device if token_mask is not None else None
        sparse = _make_sptrans_block_precision_map(
            seq_len,
            BLOCK_M,
            BLOCK_N,
            COMPUTE_BLOCK_N,
            percentage,
            pattern_params,
            device=target_device,
        )
        return pack_block_precision_map(sparse)

    direct_block_construction = (
        percentage is not None
        and pattern_type in block_percentage_patterns
        and pattern_params is not None
    )
    if token_mask is None and not direct_block_construction:
        raise ValueError(
            "token_mask may be None only for a percentage-aware block policy"
        )
    if direct_block_construction:
        target_device = token_mask.device if token_mask is not None else None
        mask2d = torch.empty(0, dtype=torch.bool, device=target_device)
        tiles = None
    else:
        mask2d = token_mask[0, 0]
        pad_m = num_m * BLOCK_M - seq_len
        pad_n = num_n * BLOCK_N - seq_len
        mask_padded = torch.nn.functional.pad(
            mask2d, (0, pad_n, 0, pad_m), value=True)

        tiles = (
            mask_padded
            .unfold(0, BLOCK_M, BLOCK_M)
            .unfold(1, BLOCK_N, BLOCK_N)
        )

    if pattern_type == "one":
        sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=mask2d.device)
    elif pattern_type == "band" and percentage is not None:
        target_band_blocks = int(round(num_n * (1.0 - percentage)))
        if target_band_blocks % 2 == 0:
            target_band_blocks = max(1, target_band_blocks - 1)
        target_band_blocks = max(1, min(target_band_blocks, num_n))

        sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=mask2d.device)
        for i in range(num_m):
            center = i
            start_block = max(0, center - target_band_blocks // 2)
            end_block = min(num_n, center + target_band_blocks // 2 + 1)
            sparse[i, start_block:end_block] = False
    elif pattern_type == "global" and percentage is not None and pattern_params is not None:
        target_global_block_cols = int(round(num_n * (1.0 - percentage)))
        target_global_block_cols = max(1, min(target_global_block_cols, num_n))

        key_pos = pattern_params.get("key_pos", list(range(min(4, seq_len))))
        global_block_cols = set()
        for pos in key_pos:
            if 0 <= pos < seq_len:
                block_col = pos // BLOCK_N
                if block_col < num_n:
                    global_block_cols.add(block_col)

        while len(global_block_cols) < target_global_block_cols and len(global_block_cols) < num_n:
            for col in range(num_n):
                if col not in global_block_cols:
                    global_block_cols.add(col)
                    break

        global_block_cols = sorted(list(global_block_cols))[:target_global_block_cols]

        sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=mask2d.device)
        for i in range(num_m):
            for block_col in global_block_cols:
                if 0 <= block_col < num_n:
                    sparse[i, block_col] = False
    elif pattern_type == "row_rand" and percentage is not None and pattern_params is not None:
        target_rand_blocks_per_row = int(round(num_n * (1.0 - percentage)))
        target_rand_blocks_per_row = max(1, min(target_rand_blocks_per_row, num_n))

        sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=mask2d.device)
        torch.manual_seed(42)
        for i in range(num_m):
            rand_indices = torch.randperm(num_n, device=mask2d.device)[:target_rand_blocks_per_row]
            sparse[i, rand_indices] = False
    elif pattern_type == "align_sparse" and percentage is not None and pattern_params is not None:
        target_aligned_blocks_per_row = int(round(num_n * (1.0 - percentage)))
        target_aligned_blocks_per_row = max(1, min(target_aligned_blocks_per_row, num_n))

        sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=mask2d.device)
        for i in range(num_m):
            start_block = max(0, num_n - target_aligned_blocks_per_row)
            sparse[i, start_block:] = False
    elif pattern_type == "bigbird" and percentage is not None and pattern_params is not None:
        target_high_precision_blocks = int(round(num_m * num_n * (1.0 - percentage)))
        target_high_precision_blocks = max(1, min(target_high_precision_blocks, num_m * num_n))

        sparse = torch.ones((num_m, num_n), dtype=torch.bool, device=mask2d.device)

        num_band_block = pattern_params.get("num_band_block", 3)
        num_global_block = pattern_params.get("num_global_block", 2)
        num_rand_block = pattern_params.get("num_rand_block", 3)

        target_band_blocks_per_row = int(round(num_n * (1.0 - percentage) * 0.5))
        target_band_blocks_per_row = max(1, min(target_band_blocks_per_row, num_n))
        band_width_blocks = target_band_blocks_per_row
        if band_width_blocks % 2 == 0:
            band_width_blocks += 1
        band_width_blocks = min(band_width_blocks, num_n)

        for i in range(num_m):
            center_col = min(i, num_n - 1)
            start_block = max(0, center_col - band_width_blocks // 2)
            end_block = min(num_n, center_col + band_width_blocks // 2 + 1)
            sparse[i, start_block:end_block] = False

        num_global_blocks = min(num_global_block, num_n)
        global_block_cols = list(range(0, num_global_blocks))
        for col in global_block_cols:
            sparse[:, col] = False
            if col < num_m:
                sparse[col, :] = False

        num_rand_blocks_per_row = min(num_rand_block, num_n)
        torch.manual_seed(42)
        for i in range(num_m):
            rand_indices = torch.randperm(num_n, device=mask2d.device)[:num_rand_blocks_per_row]
            sparse[i, rand_indices] = False

        current_high_precision = (~sparse).sum().item()
        if current_high_precision < target_high_precision_blocks:
            low_precision_indices = torch.nonzero(sparse, as_tuple=False)
            num_to_flip = target_high_precision_blocks - current_high_precision
            if num_to_flip > 0 and len(low_precision_indices) > 0:
                flip_indices = torch.randperm(len(low_precision_indices), device=mask2d.device)[:num_to_flip]
                for idx in flip_indices:
                    row, col = low_precision_indices[idx]
                    sparse[row, col] = False
        elif current_high_precision > target_high_precision_blocks:
            high_precision_indices = torch.nonzero(~sparse, as_tuple=False)
            num_to_flip = current_high_precision - target_high_precision_blocks
            if num_to_flip > 0 and len(high_precision_indices) > 0:
                flip_indices = torch.randperm(len(high_precision_indices), device=mask2d.device)[:num_to_flip]
                for idx in flip_indices:
                    row, col = high_precision_indices[idx]
                    sparse[row, col] = True
    else:
        if pattern_type == "causal":
            sparse = tiles.any(dim=(2, 3))
        else:
            sparse = (~tiles).any(dim=(2, 3))

        if percentage is not None and pattern_type not in ["one", "causal"]:
            sparse = _apply_percentage_control(sparse, percentage)

    if (
        percentage is not None
        and pattern_type
        in {
            "band",
            "global",
            "row_rand",
            "align_sparse",
            "bigbird",
        }
    ):
        weights = _causal_compute_tile_weights(
            seq_len,
            BLOCK_M,
            BLOCK_N,
            COMPUTE_BLOCK_N,
            device=sparse.device,
        )
        sparse = _retarget_weighted_int8_fraction(
            sparse,
            weights,
            percentage,
            seed=42,
        )

    return pack_block_precision_map(sparse)
