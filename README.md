# TileMix: Tile-Centric Mixed-Precision Attention

TileMix is an inference-time attention kernel for long-context LLMs. It treats **precision as a spatial routing decision over hardware-aligned score-tile groups**: each legal tile group is routed to either an FP16 or INT8 score-computation path inside one fused attention kernel, while both paths update the same online-softmax state. TileMix preserves dense attention connectivity and requires no retraining.

<p align="center">
  <img src="assets/framework.png" width="100%" alt="TileMix framework">
</p>

This repository contains the implementation accompanying **“TileMix: Tile-Centric Mixed-Precision Attention for LLM Inference Acceleration.”** The released code includes the Triton attention kernels, routing-map construction, and Hugging Face integrations for LLaMA- and Qwen-family models.

## Usage

Use a CUDA-enabled environment with PyTorch, Triton, and Hugging Face Transformers. Run the following commands from the repository root.

### LLaMA

```bash
python examples/apply_tilemix_llama.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --pattern sptrans \
  --percentage 0.50
```

Or integrate TileMix directly:

```python
from models.tilemix_llama import (
    inject_tilemix_into_llama,
    configure_tilemix_attention_pattern,
)

model = inject_tilemix_into_llama(model)
model = configure_tilemix_attention_pattern(
    model,
    pattern_type="sptrans",
    percentage=0.50,
)
```

### Qwen

```bash
python examples/apply_tilemix_qwen.py \
  --model Qwen/Qwen2-7B-Instruct \
  --pattern sptrans \
  --percentage 0.50
```

Or integrate TileMix directly:

```python
from models.tilemix_qwen import (
    inject_tilemix_into_qwen,
    configure_tilemix_attention_pattern,
)

model = inject_tilemix_into_qwen(model)
model = configure_tilemix_attention_pattern(
    model,
    pattern_type="sptrans",
    percentage=0.50,
)
```

`percentage` specifies the target fraction of legal tile groups routed to INT8. The implementation also supports the routing layouts used in the paper, including `zero`, `one`, `band`, `global`, `row_rand`, `align_sparse`, `bigbird`, and `sptrans`.
