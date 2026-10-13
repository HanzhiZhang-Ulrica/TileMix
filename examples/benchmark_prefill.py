import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.tilemix_llama import (
    PrefillCache,
    prepare_llama_prefill,
    configure_tilemix_attention_pattern,
    inject_tilemix_into_llama,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.batch_size, args.warmups, args.repetitions) < 1 or args.input_length < 64:
        parser.error("batch size, warmups and repetitions must be positive; input length must be at least 64")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)

    os.environ.update(
        TILEMIX_SCORE_ONLY_PV="1",
        TILEMIX_BITMASK_ROUTING_MODE="mixed",
        TILEMIX_FORCE_CONFIG="128,64,8,2,0",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map={"": "cuda:0"},
    ).eval()
    inject_tilemix_into_llama(model)
    configure_tilemix_attention_pattern(
        model, pattern_type="sptrans", percentage=0.50, stride=128, c=32,
    )
    prepare_llama_prefill(model)
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        100, model.config.vocab_size, (args.batch_size, args.input_length), device="cuda",
    )

    def call():
        return model(
            input_ids=input_ids, past_key_values=PrefillCache(),
            use_cache=True, logits_to_keep=1,
            output_attentions=False, output_hidden_states=False, return_dict=True,
        )

    samples = []
    with torch.inference_mode():
        for _ in range(args.warmups):
            output = call()
            del output
        torch.cuda.synchronize()
        for _ in range(args.repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            del output
        output = call()
        if not torch.isfinite(output.logits).all().item():
            raise RuntimeError("Non-finite output")
    median_ms = statistics.median(samples)
    print(json.dumps({
        "scope": "prefill_with_kv_cache_and_last_token_logits",
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "seed": args.seed,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "median_ms": median_ms,
        "k_tokens_per_second": args.batch_size * args.input_length / median_ms,
        "samples_ms": samples,
    }, indent=2))


if __name__ == "__main__":
    main()
