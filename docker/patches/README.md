# Patches

This directory contains the 43 patches applied on top of vLLM 0.28.0.

The critical ones (in dependency order):

| # | File | Purpose |
|---|---|---|
| 1 | 001-lm_head-int8.patch | Requantize lm_head + embed_tokens to INT8 (saves 2.6GB) |
| 2 | 002-qwen35-embed-quant.patch | Route embedding to vLLM's quantized kernel |
| 3 | 003-mamba-fp16-state.patch | 16-bit recursive state (37→64 concurrent seats) |
| 4 | 004-int8-act.patch | **INT4 weights × INT8 activations** (4× MMA rate) |
| 5 | 005-int8-negative-scales.patch | Fix AutoRound negative scale bug |
| 6 | 006-draft-vocab-40k.patch | 40K draft vocab (optional, English-only) |
| 7 | 007-split-kv-triton.patch | Split-KV verification attention (57µs→23µs) |
| 8 | 008-unsorted-sampler.patch | Skip full-vocab sort in top-k/top-p |
| 9 | 009-dflash2-backport.patch | DFlash2 block drafter (TP=1 only) |
| 10 | 010-hybrid-prefix-cache.patch | Prefix caching for hybrid SSM+attention |
| 11-43 | ... | Bug fixes, robustness, serving improvements |

The remaining 33 patches include:
- MTP spec decode fixes (draft token handling, accept length)
- FP8 KV cache correctness
- Qwen3 hybrid attention (linear+full) fixes
- Auto tool choice / function calling
- Torch.compile compatibility
- OOM fixes for long context
- GPU memory leak fixes
- Concurrency improvements
- API compatibility fixes

Full patch files are in this directory. Apply in numerical order with:
```bash
for p in *.patch; do patch -p1 < "$p"; done
```
