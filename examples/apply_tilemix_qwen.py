from pathlib import Path
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.tilemix_qwen import (
    configure_tilemix_attention_pattern,
    inject_tilemix_into_qwen,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--pattern", default="sptrans")
    parser.add_argument("--percentage", type=float, default=0.50)
    parser.add_argument("--prompt", default="Summarize the purpose of mixed-precision attention in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    inject_tilemix_into_qwen(model)
    configure_tilemix_attention_pattern(
        model,
        pattern_type=args.pattern,
        percentage=args.percentage,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
