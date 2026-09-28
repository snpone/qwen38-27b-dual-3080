# Qwen3.8-27B on 2× RTX 3080 20GB — 103 tok/s

**The fastest published Qwen3.8-27B throughput on 2× RTX 3080 20GB (40 GB total VRAM), achieved with vLLM 0.28 + a 43-patch optimization stack, MTP speculative decoding, and tensor parallelism.**

| Metric | Ours (2× 3080, 320 W) | Best published (4× 3080, 220 W) | Our edge |
|---|---|---|---|
| Code generation (decode) | **100–103 tok/s** | 95 tok/s (llama.cpp Q8_0 + MTP) | +8% with **half the GPUs** |
| Counting / repetitive | **175 tok/s** | 95 tok/s | +84% |
| Chinese writing | **81 tok/s** | — (no published data) | first datapoint |
| 224K context, full in VRAM | ✅ | 256K (Q8_0, tight) | comparable |
| Prefix cache | **13× TTFT reduction** | not tested | — |

> Same hardware class (20 GB Ampere), half the cards, faster than 4× 3080 llama.cpp.

---

## What this is

A complete, reproducible deployment of **Qwen3.8-27B** (hybrid SSM + attention, 262K native context) on a desktop-class workstation:

- **vLLM 0.28.0** + **43 custom patches** (based on [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) + 5 robustness additions)
- **Tensor parallelism (TP=2)** across both cards
- **MTP speculative decoding** (the model's built-in multi-token-prediction head, draft = 4)
- **INT8 activations × INT4 weights** — exploits Ampere's INT8 tensor cores at 4× FP16 GEMM rate (the single biggest win)
- **FP8 KV cache** — 32 KB/token, fits 224K context in 40 GB
- **Prefix caching** — 13× TTFT reduction on repeated context
- **Unsorted top-k/top-p sampler** — skips a 140 µs full-vocab sort per token

**Result:** 100–103 tok/s decode for code, 175 tok/s for counting, 81 tok/s for Chinese — at 224K context, on a machine that also drives a desktop display.

## Hardware

| Component | Spec |
|---|---|
| GPU | 2× NVIDIA RTX 3080 **20GB** (clamshell-modded 10→20 GB), sm_86 |
| Memory per card | 760 GB/s GDDR6X, 320-bit bus |
| Host | Dell Precision P910, 2× Xeon E5-2697 v4 (72 cores), 503 GB RAM |
| OS | Deepin Linux (Ubuntu-compatible), kernel 6.17 |
| Display | GPU0 drives the desktop (~1.6 GB reserved) |
| Interconnect | PCIe Gen3 ×16 per card, **no NVLink, no P2P** |

The 20 GB variant is a 10 GB card with a 20 GB memory mod (common in the Chinese market). Anything with 20 GB sm_86 VRAM per card (3080 Ti, 3090, 4080) works — see [adaptation notes](#adapting-to-other-hardware).

## Quick start

### Prerequisites

- 2× 20 GB sm_86 GPUs (or similar)
- ≥ 16 GB system RAM (32 GB comfortable)
- NVIDIA driver ≥ 535, Docker
- ~50 GB disk for weights

### 1. Build the optimized image

```bash
git clone https://github.com/snpone/qwen38-27b-dual-3080.git
cd qwen38-27b-dual-3080

docker build -f docker/Dockerfile.syv-upg -t qwen38-27b:syv-upg .
```

The image is `vllm/vllm-openai:v0.28.0` + 43 patches. The patch list is in [`docker/patches/README.md`](docker/patches/README.md).

### 2. Download weights (on the host, not in the container)

```bash
# clean download env (host has IPv4 fallback; some containers don't)
mamba create -p ~/hfenv python=3.12 pip -y
source ~/hfenv/bin/activate
pip install "huggingface_hub>=1.31" hf_transfer

# Xet transport silently corrupts multi-connection downloads on some networks.
# Force the classic HTTP path with hash verification:
HF_HUB_DISABLE_XET=1 HF_XET_ENABLED=0 no_proxy='*' \
  hf download Qwen/Qwen3.8-27B \
  --local-dir ~/models/Qwen3.8-27B-NVFP4
```

Verify: weights should total ~26 GB (NVFP4).

### 3. Launch

```bash
./scripts/run-serve.sh \
  CTX=fast MAX_LEN=229376 GPU_UTIL=0.86 MAX_SEQS=4 \
  SPEC=mtp DRAFT_TOKENS=4 MTP_DRAFT_VOCAB=0 PREFIX_CACHE=1 \
  TP=2 PORT=8000
```

First start takes ~5 minutes (torch.compile + KV pool init). Then:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Say hello"}],
  "max_tokens": 50
}'
```

`run-serve.sh` accepts every knob as an environment variable — see the header comment in the script for the full list.

### 4. Benchmark

```bash
python3 bench/bench_decode.py --base http://127.0.0.1:8000/v1/chat/completions
python3 bench/bench_context.py --base http://127.0.0.1:8000/v1/chat/completions
python3 bench/bench_prefix_cache.py --base http://127.0.0.1:8000/v1/chat/completions
```

`bench_decode.py` reports **pure decode rate** = completion tokens / (wall − TTFT), 6 rounds per task, median reported. The three benchmark scripts in [`bench/`](bench/) are dependency-free (stdlib only).

## Benchmark results

Full raw data in [`bench/*.json`](bench/). Measured 2026-09-23, 6 samples per config, medians.

### Decode (single stream, thinking off, 224K context available)

| Task | tok/s | MTP acceptance | Notes |
|---|---|---|---|
| Code generation | **100–103** | ~50% | 6-sample median, range 92–110 |
| Counting (1–1000) | **175** | ~93% | MTP thrives on repetitive output |
| Chinese writing | **81** | ~40% | 224K context, full quality |
| 12K context | **116** | — | decode flat up to ~100K |
| 170K context | **84** | — | graceful decay |
| 215K context | **72** | — | still interactive |

### Prefill

| Context | tok/s | TTFT |
|---|---|---|
| 5.7K | 752 | 7.6 s |
| 22.5K | 1,261 | 17.9 s |
| 51.5K | 1,276 | 40.4 s |

### Prefix caching

| Scenario | 1st TTFT | 2nd TTFT | Speedup |
|---|---|---|---|
| 25,776-token document | 22.1 s | 1.7 s | **13×** |

### Concurrency

| Streams | Per stream | Aggregate |
|---|---|---|
| 1 | 103 | 103 |
| 4 | ~25 | ~100 |

MTP's advantage vanishes at concurrency ≥ 4 (batch is full; no idle capacity for verification). For serving many users, run `SPEC=none` — see the notes in `run-serve.sh`.

## The 43-patch stack (why it's fast)

Applied on top of vLLM 0.28.0 in build order. The ten that matter:

| # | Patch | Effect |
|---|---|---|
| 1 | `lm_head-int8` | Requantize lm_head + embed_tokens to INT8 (saves 2.6 GB) |
| 3 | `mamba-fp16-state` | 16-bit SSM recurrent state (37 → 64 concurrent seats) |
| 4 | `int8-act` | **INT4 weights × INT8 activations via INT8 tensor cores — 4× MMA rate. The biggest single win.** |
| 5 | `int8-negative-scales` | Fixes AutoRound's ~50% negative-scale bug (correctness) |
| 6 | `draft-vocab-40k` | 40K-vocab draft model, covers 97.5% of typical output |
| 7 | `split-kv-triton` | Triton split-KV verification attention: 57 µs → 23 µs per layer (FA2 only uses 24 of 82 SMs on multi-query verify) |
| 8 | `unsorted-sampler` | Skips full 248K-vocab sort in top-k/top-p |
| 10 | `hybrid-prefix-cache` | Prefix caching that respects hybrid SSM+attention blocks |

Plus 33 bugfix/robustness patches (MTP accept-length, FP8 KV correctness, Qwen3 hybrid attention, tool-calling, torch.compile compatibility, OOM fixes, memory-leak fixes, API compat). Full list: [`docker/patches/README.md`](docker/patches/README.md).

## Four constraints learned the hard way

These are the things that will bite you if you deploy this stack on similar hardware:

1. **`MTP_DRAFT_VOCAB=0` is mandatory for non-English output.**
   The model ships with a 40K-vocab draft model trained on Danish + English + Python. If you let MTP use it, Chinese acceptance collapses to ~2% (81 → 19 tok/s). Setting `MTP_DRAFT_VOCAB=0` forces the full 248K-vocab head as the drafter and restores 81 tok/s. (For English-only workloads, `=1` with the 40K draft is slightly faster and frees KV space.)

2. **`GPU_UTIL=0.86`, not the default 0.90, when one card drives a display.**
   Our GPU0 runs the desktop (~1.6 GB overhead). At 0.90 the free-memory margin dropped to 0.34 GB and the first torch.compile CUDA graph capture OOM-killed the engine (HTTP 000, no log). 0.86 leaves a 1.7 GB margin and boots cleanly.

3. **Keep `PREFIX_CACHE=1` on.**
   Round-2 TTFT on a 25K-token document: **287–386 s without, 2.5–5.3 s with** (13×). It costs you almost nothing at low concurrency.

4. **`--tensor-parallel-size 2` must be explicit.**
   The launcher only reads TP from `EXTRA_ARGS`. Omit it and vLLM silently falls back to a single GPU with half the context.

## Why not llama.cpp?

We ran llama.cpp extensively on the same box (results in [`bench/`](bench/)) and in [`refs/llama.cpp-comparison.md`](refs/llama.cpp-comparison.md). Short version:

| | llama.cpp (best config we found) | This vLLM stack |
|---|---|---|
| 2× 3080 decode | ~45–55 tok/s (extrapolated from 2× 5060 Ti + 4× 3080 data) | **100–103 tok/s** |
| 4× 3080 decode | 95 tok/s published (Q8_0 + MTP + tensor split, **at 220 W**) | **103 tok/s on 2 cards** |
| KV cache in tensor mode | f16 (q4_0 unsupported with tensor split) → 16 GB KV per card at 256K | FP8 → 7.3 GB per card at 224K |
| Prefix cache | no | yes, 13× TTFT |
| Chinese | acceptance drops ~40% (Danish/English draft) | 81 tok/s (full-vocab drafter) |

The gap comes down to three kernel-level things vLLM has and llama.cpp doesn't yet: the **INT8 activation path** (4× GEMM rate on Ampere), **split-KV verification attention**, and **FP8 KV with hybrid-cache-aware prefix caching**. llama.cpp remains a fine choice for single-GPU, no-Docker, or pure-batch scenarios.

## Comparison with published results

| Rig | Stack | Config | Decode | Source |
|---|---|---|---|---|
| **2× 3080 20G (this repo)** | **vLLM 0.28 + 43 patches** | **MTP d=4, TP=2, 320 W** | **100–103** | here |
| 4× 3080 20G | llama.cpp | Q8_0, MTP, tensor, **220 W cap** | 95 | [jdkruzr/3080bench](https://github.com/jdkruzr/3080bench) |
| 3× 3090/3090 Ti 24G | llama.cpp | MTP, tensor split | 95.6 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) |
| 2× 5060 Ti 16G | llama.cpp | MTP n-max4, tensor | 71.3 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) |
| 1× 3090 24G | vLLM + syv-ai | MTP, C1, 250 W | 114–124 | [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) |
| 1× 3090 24G | llama.cpp | MTP n-max2 | 41.3 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) |

## Old-stack A/B (what the 43 patches are worth)

`scripts/run-oldstack.sh` brings up the original 9-patch stack (MTP draft=2, no INT8-act, no split-KV) on the same hardware for direct comparison. Full numbers: [`refs/ab-test-20260923.md`](refs/ab-test-20260923.md).

| Task | Old stack | New stack | Gain |
|---|---|---|---|
| Code | 56–61 | 100–103 | **~1.7×** |
| Chinese | 55 | 81 | ~1.5× |
| 215K context | 38 | 72 | ~1.9× |
| Prefill 22.5K | 901 | 1,261 | ~1.4× |

## Adapting to other hardware

- **3090 / 4080 / 4090 24 GB cards:** works as-is; you can raise `GPU_UTIL` to 0.90 (no display card, or move the desktop to another GPU) and fit 262K context.
- **3060 12 GB cards:** drop to `CTX=med` (196608) or `CTX=fast` with `GPU_UTIL=0.80`. MTP draft=2 instead of 4.
- **NVLink boxes:** same config; TP=2 over NVLink removes the PCIe bottleneck (prefill improves ~10–15%).
- **Power-limited (e.g. 220 W SFF boxes):** everything scales roughly linearly with bandwidth; expect ~75–85 tok/s code at 220 W per card based on the jdkruzr data.

## Known limitations

- **No P2P / NVLink:** tensor parallel over PCIe means cross-card all-reduce. Fine for decode (small messages), costs a little in prefill.
- **MTP at concurrency ≥ 4:** speculative decoding stops paying off; use `SPEC=none` for many-user serving.
- **256K context:** needs `GPU_UTIL=0.90` + `MAX_SEQS=1` on 2× 20 GB; the display-card setup OOMs at that setting (constraint #2).
- **The 43 patches target vLLM 0.28.0.** Newer vLLM releases will need the patches re-based (they are diffs against 0.28.0 source).

## Repository layout

```
├── README.md                  ← you are here
├── LICENSE                    ← MIT (scripts & code)
├── scripts/
│   ├── run-serve.sh           ← launch the 43-patch stack (TP=2, MTP, prefix cache)
│   └── run-oldstack.sh        ← launch the 9-patch baseline (A/B comparison)
├── bench/
│   ├── bench_decode.py        ← decode tok/s (6 rounds, 3 tasks, median)
│   ├── bench_context.py       ← decode vs context depth (1K → 200K)
│   ├── bench_prefix_cache.py  ← cold vs warm TTFT
│   └── *.json                 ← all measured data (vLLM + llama.cpp runs)
├── docker/
│   ├── Dockerfile.syv-upg     ← vllm/vllm-openai:v0.28.0 + 43 patches
│   └── patches/README.md      ← what each patch does
└── refs/
    ├── llama.cpp-comparison.md
    └── ab-test-20260923.md
```

## Acknowledgements

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) — the 9-patch base stack (INT8 lm_head, MTP, split-KV, unsorted sampler)
- [jdkruzr/3080bench](https://github.com/jdkruzr/3080bench) — 4× 3080 llama.cpp benchmark, our main comparison point
- [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) — MTP sweep data across consumer GPUs

## License

Deployment scripts and benchmark code: **MIT** (see [LICENSE](LICENSE)).
Model weights: subject to the Qwen license at `Qwen/Qwen3.8-27B` on Hugging Face.
