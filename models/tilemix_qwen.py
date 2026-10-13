import math
from dataclasses import dataclass
from typing import Optional, Tuple
import os

import torch
from torch import Tensor
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Qwen2Model, apply_rotary_pos_emb
)
from attention.tilemix_varlen import tilemix_varlen_attention
from attention.precision_maps import (
    gen_bigbird_pattern,
    gen_one_pattern,
    gen_zero_pattern,
    gen_causal_pattern,
    gen_band_pattern,
    gen_align_sparse_pattern,
    gen_row_rand_pattern,
    gen_global_pattern,
    gen_block_pattern,
    layout_to_causal_mask,
    make_precision_bitmask
)


@dataclass(frozen=True)
class BucketConfig:
    block_m: int
    block_n: int


def pick_block_n_for_64(seq_len: int, choices: list[int]) -> int:
    for bn in sorted(choices):
        if math.ceil(seq_len / bn) <= 64:
            return bn
    return max(choices)


def pick_mask_block_n(seqlen: int, compute_block_n: int) -> int:
    min_mask_bn = (seqlen + 63) // 64
    mask_bn = ((max(min_mask_bn, compute_block_n) + compute_block_n - 1) // compute_block_n) * compute_block_n
    return mask_bn


class TileMixqwenAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)

        self.decode_precision_cache: dict[tuple, Tensor] = {}

        nh = config.num_attention_heads
        nkv = config.num_key_value_heads
        if nh % nkv != 0:
            raise ValueError(f"num_attention_heads ({nh}) must be divisible by num_key_value_heads ({nkv})")

        repeat = nh // nkv
        self.register_buffer(
            "kv_head_mapping",
            torch.arange(nkv, dtype=torch.long).repeat_interleave(repeat)
        )
        self.nh, self.nkv = nh, nkv

        self._bucket_configs = {
            "decode": BucketConfig(64, 64),
            "short":  BucketConfig(64, 64),
            "medium": BucketConfig(64, 64),
            "long":   BucketConfig(64, 64),
            "xl":     BucketConfig(64, 64),
            "xxl":    BucketConfig(64, 64),
        }

        self._supported_ns = [64, 128, 256]

        self.available_block_sizes = [
            (config.block_m, config.block_n) for config in self._bucket_configs.values()
        ]

        additional_sizes = [
            (64, 64),
            (64, 128),
            (64, 256),
            (128, 64),
        ]
        for size in additional_sizes:
            if size not in self.available_block_sizes:
                self.available_block_sizes.append(size)

        self._block_to_candidate: dict[tuple[int,int], int] = {
            bs: i for i, bs in enumerate(self.available_block_sizes)
        }

        self.precision_map_cache = {}

        self.default_pattern_type = "one"
        self.default_pattern_params = {}

        self._prewarm_triton_kernels()

    def _choose_bucket(self, seqlen: int) -> Optional[str]:
        if seqlen <= 128:
            return "decode"
        if seqlen <= 2048:
            return "short"
        if seqlen <= 8192:
            return "medium"
        if seqlen <= 16384:
            return "long"
        if seqlen <= 32768:
            return "xl"
        return "xxl"

    def _find_nearest_available_block_size(self, seqlen: int, block_m: int, block_n: int) -> tuple[int, int]:
        best_size = None
        best_distance = float('inf')
        for available_m, available_n in self.available_block_sizes:
            if available_n > seqlen:
                continue
            distance = abs(block_m - available_m) * 0.5 + abs(block_n - available_n) * 1.5
            if distance < best_distance:
                best_distance = distance
                best_size = (available_m, available_n)
        if best_size is None:
            supported_ms = [64, 128, 256, 512]
            adj_block_m = min(supported_ms, key=lambda x: abs(x - block_m))
            adj_block_n = min(self._supported_ns, key=lambda x: abs(x - block_n))
            return adj_block_m, adj_block_n
        return best_size

    def _get_block_config(self, seqlen: int) -> BucketConfig:
        forced = os.getenv("TILEMIX_FORCE_CONFIG")
        if forced:
            try:
                parts = [p.strip() for p in forced.split(",")]
                bm = int(parts[0]); bn = int(parts[1])
                bm, bn = self._find_nearest_available_block_size(seqlen, bm, bn)
                return BucketConfig(bm, bn)
            except Exception:
                pass

        bucket = self._choose_bucket(seqlen)
        if bucket is not None:
            cfg = self._bucket_configs[bucket]
            adj_block_n = cfg.block_n
            adj_block_n = min(max(adj_block_n, 64), max(self._supported_ns))

            adj_block_m = cfg.block_m
            if seqlen >= 4096:
                adj_block_m = max(adj_block_m, 128)

            adj_block_m, adj_block_n = self._find_nearest_available_block_size(seqlen, adj_block_m, adj_block_n)
            return BucketConfig(adj_block_m, adj_block_n)

        block_n = 256 if seqlen >= 8192 else 128
        block_n = min(max(block_n, 64), max(self._supported_ns))
        block_m = 64 if seqlen < 1024 else 128
        if seqlen >= 4096:
            block_m = max(block_m, 128)
        block_m, block_n = self._find_nearest_available_block_size(seqlen, block_m, block_n)
        return BucketConfig(block_m, block_n)

    def _prewarm_triton_kernels(self):
        try:
            if not torch.cuda.is_available():
                return

            device = torch.device("cuda")
            head_dim = getattr(self.config, 'hidden_size', 4096) // self.nh

            dummy_q = torch.empty((1, self.nh, 32, head_dim), dtype=torch.float16, device=device)
            dummy_k = torch.empty((1, self.nkv, 32, head_dim), dtype=torch.float16, device=device)
            dummy_v = torch.empty((1, self.nkv, 32, head_dim), dtype=torch.float16, device=device)

            for block_m, block_n in self.available_block_sizes:
                try:
                    num_m_blocks = (32 + block_m - 1) // block_m
                    num_n_blocks = (32 + block_n - 1) // block_n
                    if num_n_blocks > 64:
                        continue

                    row_mask = torch.full(
                        (num_m_blocks,),
                        (1 << num_n_blocks) - 1,
                        dtype=torch.uint64,
                        device=device,
                    )
                    dummy_precision_map = row_mask.repeat(self.nkv)

                    _ = tilemix_varlen_attention(
                        dummy_q, dummy_k, dummy_v,
                        block_m, block_n,
                        causal=True,
                        scale=head_dim ** -0.5,
                        precision_map_all=dummy_precision_map,
                        candidates=[(block_m, block_n)],
                        flat_lengths=[dummy_precision_map.numel()],
                    )
                except Exception:
                    continue

        except Exception:
            pass

    def _get_pattern_config(self, seqlen: int):
        return (self.default_pattern_type, tuple(sorted(self.default_pattern_params.items())))

    def _generate_attention_pattern(self, seqlen: int, pattern_type: str, pattern_params: dict):
        base_params = {
            "num_query": seqlen,
            "num_key": seqlen,
            "dtype": torch.bool,
            "num_layer": 1,
            "num_head": 1
        }

        percentage = pattern_params.get("percentage", None)

        if pattern_type == "one":
            return gen_one_pattern(**base_params)
        elif pattern_type == "zero":
            return gen_zero_pattern(**base_params)
        elif pattern_type == "causal":
            return gen_causal_pattern(**base_params)
        elif pattern_type == "band":
            params = {**base_params, **pattern_params}
            percentage = params.get("percentage", None)
            band_width = params.get("band_width", None)
            params["percentage"] = percentage
            if band_width is not None:
                params["band_width"] = band_width
            return gen_band_pattern(**params)
        elif pattern_type == "global":
            params = {**base_params, **pattern_params}
            params.pop("percentage", None)
            key_pos = params.get("key_pos", list(range(min(4, seqlen))))
            key_pos = [pos for pos in key_pos if 0 <= pos < seqlen]
            params["key_pos"] = key_pos
            return gen_global_pattern(**params)
        elif pattern_type == "row_rand":
            params = {**base_params, **pattern_params}
            params.pop("percentage", None)
            num_rand = min(params.get("num_rand_block_per_row", 3), seqlen)
            params["num_rand_block_per_row"] = num_rand
            return gen_row_rand_pattern(**params)
        elif pattern_type == "align_sparse":
            params = {**base_params, **pattern_params}
            params.pop("percentage", None)
            params.setdefault("alpha", min(32, seqlen // 4))
            params.setdefault("beta", 0.1)
            return gen_align_sparse_pattern(**params)
        elif pattern_type == "bigbird":
            params = {**base_params, **pattern_params}
            params.pop("percentage", None)
            params["num_global_block"] = min(params.get("num_global_block", 2), seqlen)
            params["num_rand_block"] = min(params.get("num_rand_block", 3), seqlen)
            params.setdefault("num_band_block", min(3, seqlen))
            return gen_bigbird_pattern(**params)
        elif pattern_type == "block":
            layout = pattern_params.get("layout")
            block_size = pattern_params.get("block_size", 64)

            if layout is None:
                num_blocks = (seqlen + block_size - 1) // block_size
                layout = torch.eye(num_blocks, dtype=torch.bool)
                for i in range(1, min(3, num_blocks)):
                    layout = layout | torch.eye(num_blocks, k=i, dtype=torch.bool)
                    layout = layout | torch.eye(num_blocks, k=-i, dtype=torch.bool)

            return gen_block_pattern(layout, block_size, dtype=torch.bool)
        elif pattern_type == "layout_causal":
            layout = pattern_params.get("layout")
            block_size = pattern_params.get("block_size", 64)

            if layout is None:
                num_blocks = (seqlen + block_size - 1) // block_size
                layout = torch.ones(1, 1, num_blocks, num_blocks, dtype=torch.bool)

            return layout_to_causal_mask(layout, block_size, dtype=torch.bool)
        elif pattern_type == "sptrans":
            params = pattern_params.copy()
            stride = params.get("stride", min(128, seqlen // 4))
            c = params.get("c", min(32, stride // 4))

            mask = torch.zeros((1, 1, seqlen, seqlen), dtype=torch.bool)
            for q in range(seqlen):
                qs = q // stride
                mask[0, 0, q, qs*stride : min((qs+1)*stride, q+1)] = True
                for ks in range(qs):
                    start_idx = ks*stride + stride - c
                    end_idx = (ks+1)*stride
                    if start_idx < seqlen and end_idx <= seqlen:
                        mask[0, 0, q, start_idx:end_idx] = True
            return mask
        else:
            print(f"Warning: Unknown pattern type '{pattern_type}', falling back to full attention")
            return gen_one_pattern(**base_params)

    def _get_pattern_hash(self, pattern_type: str, pattern_params: dict) -> str:
        import hashlib
        pattern_str = f"{pattern_type}_{str(sorted(pattern_params.items()))}"
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]

    def _get_or_create_precision_map(
        self,
        seqlen: int,
        block_m: int,
        block_n: int,
        block_n_mask: int,
        pattern_type: str,
        pattern_params: tuple,
        device: torch.device,
    ):
        pattern_params_dict = dict(pattern_params)
        pattern_hash = self._get_pattern_hash(pattern_type, pattern_params_dict)
        percentage = pattern_params_dict.get("percentage", None)
        if pattern_type in ["one", "causal"]:
            percentage = None
        cache_key = (seqlen, block_m, block_n, block_n_mask, pattern_hash, percentage)

        if cache_key in self.precision_map_cache:
            cached_map = self.precision_map_cache[cache_key]
            if cached_map.device != device:
                cached_map = cached_map.to(device)
                self.precision_map_cache[cache_key] = cached_map
            return cached_map

        percentage = pattern_params_dict.get("percentage", None)
        if pattern_type in ["one", "causal"]:
            percentage = None
        if pattern_type in {"zero", "one"} or (
            percentage is not None
            and pattern_type
            in {
                "band",
                "global",
                "row_rand",
                "align_sparse",
                "bigbird",
                "sptrans",
            }
        ):
            token_mask = None
        else:
            token_mask = self._generate_attention_pattern(
                seqlen,
                pattern_type,
                pattern_params_dict,
            )

        precision_bitmask = make_precision_bitmask(
            seq_len=seqlen,
            BLOCK_M=block_m,
            BLOCK_N=block_n_mask,
            token_mask=token_mask,
            percentage=percentage,
            pattern_type=pattern_type,
            pattern_params=pattern_params_dict,
            COMPUTE_BLOCK_N=block_n,
        )

        m_blocks_expected = (seqlen + block_m - 1) // block_m
        n_mask_blocks_expected = (seqlen + block_n_mask - 1) // block_n_mask
        if n_mask_blocks_expected > 64:
            raise ValueError(
                f"64-bit precision mask requires n_mask_blocks<=64, got n_mask_blocks={n_mask_blocks_expected} "
                f"(seqlen={seqlen}, block_n_mask={block_n_mask})"
            )
        expected_numel = m_blocks_expected
        if precision_bitmask.numel() != expected_numel:
            raise ValueError(
                f"Precision bitmask length mismatch: got {precision_bitmask.numel()}, "
                f"expected {expected_numel} (m_blocks={m_blocks_expected}, n_mask_blocks={n_mask_blocks_expected})"
            )

        if not isinstance(precision_bitmask, torch.Tensor):
            raise TypeError(f"precision_bitmask must be a tensor, got {type(precision_bitmask)}")

        if precision_bitmask.numel() == 0:
            raise ValueError(f"precision_bitmask cannot be empty")

        try:
            precision_bitmask = precision_bitmask.to(device=device, dtype=torch.uint64)
        except RuntimeError as e:
            raise

        self.precision_map_cache[cache_key] = precision_bitmask

        return precision_bitmask

    def set_attention_pattern(self, pattern_type: str, **pattern_params):
        self.default_pattern_type = pattern_type
        self.default_pattern_params = pattern_params

        self.precision_map_cache.clear()

    def _ensure_cuda(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.is_cuda:
            return hidden_states
        return hidden_states.cuda()

    def _prepare_precision_map(
        self,
        seqlen: int,
        bucket_config: BucketConfig,
        block_n_mask: int,
        pattern_type: str,
        pattern_params: tuple,
        device: torch.device,
    ) -> Tensor:
        if bucket_config.block_m == self._bucket_configs["decode"].block_m and bucket_config.block_n == self._bucket_configs["decode"].block_n:
            cache_key = (
                device.type,
                bucket_config.block_m,
                bucket_config.block_n,
                block_n_mask,
                self.nh,
                self.nkv,
                seqlen,
            )
            cached = self.decode_precision_cache.get(cache_key)
            if cached is not None and cached.device == device:
                return cached

        row_bitmask = self._get_or_create_precision_map(
            seqlen,
            bucket_config.block_m,
            bucket_config.block_n,
            block_n_mask,
            pattern_type,
            pattern_params,
            device,
        )

        m_blocks_total = (seqlen + bucket_config.block_m - 1) // bucket_config.block_m
        n_mask_blocks_total = (seqlen + block_n_mask - 1) // block_n_mask
        if n_mask_blocks_total > 64:
            raise ValueError(
                f"64-bit precision mask requires n_mask_blocks<=64, got n_mask_blocks={n_mask_blocks_total} "
                f"(seqlen={seqlen}, block_n_mask={block_n_mask})"
            )
        expected_row_length = m_blocks_total
        if row_bitmask.numel() != expected_row_length:
            raise ValueError(
                f"Precision bitmask row length {row_bitmask.numel()} does not match expected {expected_row_length} "
                f"(m_blocks={m_blocks_total}, n_mask_blocks={n_mask_blocks_total})"
            )


        precision_map = row_bitmask.to(device=device, dtype=torch.uint64).repeat(self.nkv)

        if bucket_config.block_m == self._bucket_configs["decode"].block_m and bucket_config.block_n == self._bucket_configs["decode"].block_n:
            self.decode_precision_cache[cache_key] = precision_map

        return precision_map

    def _project_hidden_states(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v

    def _reshape_qkv(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch_size: int,
        seqlen: int,
        head_dim: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        q = q.view(batch_size, seqlen, self.nh, head_dim).transpose(1, 2)
        k = k.view(batch_size, seqlen, self.nkv, head_dim).transpose(1, 2)
        v = v.view(batch_size, seqlen, self.nkv, head_dim).transpose(1, 2)
        return q, k, v

    def _resolve_rotary_embeddings(
        self,
        q: Tensor,
        position_ids,
        position_embeddings: Optional[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        if position_embeddings is None:
            return self.rotary_emb(q, position_ids)
        return position_embeddings

    @staticmethod
    def _ensure_rotary_on_device(
        cos: Optional[Tensor],
        sin: Optional[Tensor],
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if cos is not None and not cos.is_cuda:
            cos = cos.to(device)
        if sin is not None and not sin.is_cuda:
            sin = sin.to(device)
        return cos, sin


    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        n_heads = self.config.num_attention_heads
        n_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // n_heads
        bsz, seqlen, _ = hidden_states.size()

        if past_key_values is None:
            past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            past_key_values = kwargs.get("past_key_value", None)
        past_kv = past_key_values

        if position_embeddings is not None and not (
            isinstance(position_embeddings, tuple) and len(position_embeddings) == 2
        ):
            attention_mask = position_embeddings
            position_embeddings = kwargs.get("position_embeddings", None)

        hidden_states = self._ensure_cuda(hidden_states)
        target_device = hidden_states.device

        if seqlen < 16:
            return super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        bucket_config = self._get_block_config(seqlen)
        block_m = bucket_config.block_m
        block_n = bucket_config.block_n
        block_n_mask = pick_mask_block_n(seqlen, block_n)


        pattern_type, pattern_params = self._get_pattern_config(seqlen)

        bucket_name = self._choose_bucket(seqlen)
        if bucket_name is not None:
            block_size = (bucket_config.block_m, bucket_config.block_n)
            if block_size not in self._block_to_candidate:
                raise ValueError(
                    f"Block size {(bucket_config.block_m, bucket_config.block_n)} for bucket {bucket_name} "
                    f"is not in available_block_sizes. "
                    f"Available sizes: {self.available_block_sizes}"
                )

        precision_map_flat = self._prepare_precision_map(
            seqlen,
            bucket_config,
            block_n_mask,
            pattern_type,
            pattern_params,
            target_device,
        )
        m_blocks_total = (seqlen + bucket_config.block_m - 1) // bucket_config.block_m
        n_mask_blocks_total = (seqlen + block_n_mask - 1) // block_n_mask
        if n_mask_blocks_total > 64:
            raise ValueError(
                f"64-bit precision mask requires n_mask_blocks<=64, got n_mask_blocks={n_mask_blocks_total} "
                f"(seqlen={seqlen}, block_n_mask={block_n_mask})"
            )
        expected_pm = self.nkv * m_blocks_total
        if precision_map_flat.numel() != expected_pm:
            precision_map_flat = self._prepare_precision_map(
                seqlen,
                bucket_config,
                block_n_mask,
                pattern_type,
                pattern_params,
                target_device,
            )
            if precision_map_flat.numel() != expected_pm:
                raise ValueError(
                    f"precision_map length mismatch after regen: got {precision_map_flat.numel()}, "
                    f"expected {expected_pm} (Hk={self.nkv} × m_blocks={m_blocks_total}, "
                    f"block_m={bucket_config.block_m}, block_n={bucket_config.block_n}, block_n_mask={block_n_mask}, seqlen={seqlen})"
                )

        q, k, v = self._project_hidden_states(hidden_states)
        q = q.to(target_device)
        k = k.to(target_device)
        v = v.to(target_device)
        q, k, v = self._reshape_qkv(q, k, v, bsz, seqlen, head_dim)

        if position_embeddings is not None and isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
            cos, sin = position_embeddings
        else:
            position_ids = kwargs.get("position_ids", None)
            cos, sin = self._resolve_rotary_embeddings(q, position_ids, position_embeddings)
        cos, sin = self._ensure_rotary_on_device(cos, sin, target_device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(
                k,
                v,
                self.layer_idx,
                cache_kwargs={"sin": sin, "cos": cos, "cache_position": cache_position},
            )

        assert q.shape[1] == self.nh, f"Q heads {q.shape[1]} != nh {self.nh}"
        assert k.shape[1] == self.nkv, f"K heads {k.shape[1]} != nkv {self.nkv}"
        assert v.shape[1] == self.nkv, f"V heads {v.shape[1]} != nkv {self.nkv}"

        target_dtype = q.dtype
        q = q.to(dtype=target_dtype, device=target_device).contiguous()
        k = k.to(dtype=target_dtype, device=target_device).contiguous()
        v = v.to(dtype=target_dtype, device=target_device).contiguous()

        if not all(tensor.is_cuda for tensor in (q, k, v, precision_map_flat)):
            raise RuntimeError(
                "All tensors must be on CUDA for Triton. "
                f"q={q.device}, k={k.device}, v={v.device}, precision_map={precision_map_flat.device}"
            )


        num_n_blocks = (seqlen + block_n_mask - 1) // block_n_mask
        try:
            valid_mask = (torch.tensor(1, dtype=torch.uint64, device=precision_map_flat.device) << num_n_blocks) - 1
            invalid_bits = precision_map_flat & ~valid_mask
            if torch.any(invalid_bits != 0):
                precision_map_flat = precision_map_flat & valid_mask
            highest_bit = (torch.tensor(1, dtype=torch.uint64, device=precision_map_flat.device) << (num_n_blocks - 1))
            if torch.all((precision_map_flat != 0) & ((precision_map_flat & (precision_map_flat - 1)) == 0) & (precision_map_flat == highest_bit)):
                precision_map_flat = valid_mask.expand_as(precision_map_flat)
        except Exception:
            pass

        try:
            attn_output = tilemix_varlen_attention(
                q,
                k,
                v,
                block_m,
                block_n,
                block_n_mask=block_n_mask,
                causal=True,
                scale=head_dim ** -0.5,
                precision_map_all=precision_map_flat,
                candidates=[(bucket_config.block_m, bucket_config.block_n)],
                flat_lengths=[precision_map_flat.numel()],
            )
        except Exception as e:
            raise

        out = attn_output.transpose(1, 2).reshape(bsz, seqlen, -1)
        out = self.o_proj(out)

        return out, None


def inject_tilemix_into_qwen(model: Qwen2Model):
    root = model.model if hasattr(model, "model") else model

    for layer in root.layers:
        orig = layer.self_attn
        new_attn = TileMixqwenAttention(orig.config, orig.layer_idx)
        first_param = next(orig.parameters())
        target_device = first_param.device
        target_dtype = first_param.dtype
        new_attn = new_attn.to(device=target_device, dtype=target_dtype)

        try:
            orig_state = orig.state_dict()
            new_state = new_attn.state_dict()

            for key, value in orig_state.items():
                if key in new_state:
                    try:
                        new_state[key].copy_(value)
                    except Exception as e:
                        pass

        except Exception as e:
            try:
                new_attn.load_state_dict(orig.state_dict(), strict=False)
            except Exception as e2:
                pass

        layer.self_attn = new_attn

    model.use_tilemix = True

    return model


def configure_tilemix_attention_pattern(
    model: Qwen2Model,
    pattern_type: str = "one",
    **pattern_params
):
    root = model.model if hasattr(model, "model") else model
    for layer in root.layers:
        if isinstance(layer.self_attn, TileMixqwenAttention):
            layer.self_attn.set_attention_pattern(
                pattern_type,
                **pattern_params
            )
    return model
