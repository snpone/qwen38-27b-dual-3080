# llama.cpp vs vLLM on 2× RTX 3080 20GB for Qwen3.8-27B

## Summary

| | llama.cpp | vLLM + 43 patches (ours) |
|---|---|---|
| 2× 3080 decode (best) | ~45-55 tok/s (extrapolated) | **100-103 tok/s** |
| 4× 3080 decode (published) | 95 tok/s @ 220W | **103 tok/s (2 cards!)** |
| 3× 3090 decode (published) | 95.6 tok/s | 103 tok/s (2× 3080!) |
| 1× 3090 decode (published) | 41.3 tok/s | — |
| KV cache | f16 (tensor mode) / q4_0 | FP8 (32KB/token) |
| Context in 40GB total | ~256K (Q8_0, tight) | 224K comfortable |
| Prefix caching | No | **Yes (13× TTFT)** |
| Chinese | Drops ~40% (Danish/EN draft) | 81 tok/s (full draft vocab) |
| INT8 tensor core | No | **Yes (4× MMA rate)** |
| Split-KV attention | No | **Yes (23µs/layer)** |

## Why llama.cpp is slower on this hardware

### 1. No INT8 tensor core path

The single biggest differentiator. On Ampere (sm_86), INT8 tensor cores run at
**4× the rate of FP16 GEMM**. Our `INT8_ACT` patch converts activations to INT8
before the matmul, leveraging this. llama.cpp's default kernels use FP16 GEMM
for weight×activation, missing this 4× speedup entirely.

For decode (batch=1), the GEMM is memory-bound, but the **compute intensity**
still matters for the attention score computation and the lm_head projection.
The 248K-vocab lm_head is a 2560×248128 matmul — INT8 vs FP16 makes a real
difference here.

### 2. No split-KV for multi-query verification

With MTP speculative decoding, the verify step processes K+1 queries per
attention layer (K = draft tokens). llama.cpp's FA2 kernel only uses 24 of 82
SMs for this (the rest idle). Our Triton split-KV kernel distributes across
all SMs: **57µs → 23µs per layer**.

### 3. KV cache format

llama.cpp requires f16 KV in tensor-split mode (q4_0 doesn't work with
`--split-mode tensor`). f16 KV = 64KB/token → 256K context needs 16GB KV
per card. Our FP8 KV = 32KB/token → 224K context needs 7.3GB KV per card.

This means we fit more context in the same 40GB total, or can run at higher
gpu_memory_utilization.

### 4. Draft vocabulary

The 40K draft vocab in the model was trained on Danish+English+Python.
With `MTP_DRAFT_VOCAB=0` (full 248K vocab), Chinese acceptance goes from
~2% to ~40%. This is a **vLLM-specific fix** — llama.cpp doesn't have
this knob.

## When llama.cpp might be better

- **Batch inference**: llama.cpp's continuous batching is simpler. For pure
  batch jobs (no interactive users), the gap narrows.
- **Single GPU**: On 1× 3090 24G, llama.cpp Q8_0 + MTP does 41.3 tok/s.
  Our vLLM on the same card would be limited by the same memory ceiling.
  The gap is smaller with one GPU.
- **No Docker**: If you don't want to run containers, llama.cpp is a single
  binary. Our vLLM stack requires the patched Docker image.

## Our llama.cpp results (for reference)

| Test | Config | Result |
|---|---|---|
| 12.8K ctx, MTP, layer split | Q6_K, n-max2 | 41.9 t/s decode |
| 248K ctx, MTP, layer split | Q6_K, n-max2 | 19.6 t/s decode |
| 256K ctx, no MTP, layer | Q6_K, KV q4_0 | 27.3 t/s decode |
| 256K ctx, no MTP, tensor | Q8_0, KV q4_0 | 39.0 t/s decode |
| 256K ctx, MTP, tensor | Q8_0, KV q4_0 | 69.3 t/s decode |
| 3072K ctx (6× RAM offload) | Q4_K, KV q4_0 | 13.2 t/s decode |

Note: our llama.cpp tests used `--split-mode layer` (default), not `--split-mode tensor`.
Switching to tensor split would improve llama.cpp by ~68% (based on published
2× 5060 Ti data: layer 22.1 → tensor 37.1). Even then, llama.cpp would be
~60-70 tok/s on 2× 3080, still well below our vLLM's 103.
