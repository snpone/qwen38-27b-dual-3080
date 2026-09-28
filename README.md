# Qwen3.8-27B on Dual RTX 3080 20GB — 103 tok/s Real-World Benchmark

**The fastest published Qwen3.8-27B numbers on 2× 3080 20GB (40GB total VRAM) — using vLLM + 43-patch optimization stack + MTP speculative decoding + tensor parallelism.**

| Metric | Ours (2× 3080) | Best published (4× 3080) | Ours wins by |
|---|---|---|---|
| Code generation | **103 tok/s** | 95 tok/s (llama.cpp, 220W) | +8% with half the GPUs |
| Count/repeat | **175 tok/s** | 95 tok/s | +84% |
| Chinese writing | **81 tok/s** | — | first published data point |
| 224K context | ✅ fits | 256K (Q8_0, tight) | comparable |
| Prefix cache | 13× TTFT reduction | not tested | — |

> **Same hardware class (20GB Ampere), half the cards, faster than 4× 3080 llama.cpp.**

## TL;DR

We deploy Qwen3.8-27B (hybrid SSM+attention, 262K native context) on **2× RTX 3080 20GB** with:

- **vLLM 0.28.0** + **43 custom patches** (from [syv-ai](https://github.com/syv-ai/qwen38-27b-rtx3090) + 5 robustness additions)
- **Tensor Parallelism (TP=2)** — splits matmuls across both cards
- **MTP speculative decoding** — uses the model's built-in multi-token-prediction head
- **INT8 tensor core inference** — 4× MMA rate on Ampere (the killer patch)
- **FP8 KV cache** — 32 KB/token, fits 224K context in 40GB
- **Prefix caching** — 13× TTFT reduction on repeated contexts

Result: **100-103 tok/s** decode for code tasks, **175 tok/s** for counting, **81 tok/s** for Chinese writing — all at 224K context on a workstation with a desktop display.

## Why this beats 4× 3080 llama.cpp

The 4× 3080 [jdkruzr/3080bench](https://github.com/jdkruzr/3080bench) result (95 tok/s) is impressive but runs at a **220W power cap** (container restriction) and uses **llama.cpp's default kernels**. Our stack:

1. **INT8 activation × INT4 weights** (`INT8_ACT` patch) — uses Ampere's INT8 tensor cores at 4× rate vs FP16 GEMM. This is the single biggest win.
2. **Split-KV Triton verification attention** — 23µs/layer vs 57µs (FA2 only uses 24/82 SMs on multi-query verification steps)
3. **Unsorted top-k/top-p sampler** — skips full 248K-vocab sort (140µs → negligible)
4. **Full 320W power headroom** — 4× 3080 bench was power-capped; our cards run at stock 320W

## Hardware

| Component | Spec |
|---|---|
| GPU | 2× NVIDIA RTX 3080 20GB (clamshell-modded 10→20GB, sm_86) |
| Memory | 760 GB/s per card, 320-bit bus, GDDR6X |
| Host | Dell P910, 2× E5-2697 v4 (72 cores), 503GB RAM |
| OS | Deepin Linux (Ubuntu-compatible), kernel 6.17 |
| Display | GPU0 drives desktop (~1.6GB reserved) |
| Interconnect | PCIe Gen3 ×16 per card, no NVLink, no P2P |

## Quick Start

### Prerequisites

- 2× RTX 3080 20GB (or similar 20GB sm_86 cards: 3080 Ti, 3090, 3060 12GB with reduced context)
- 192GB+ system RAM
- NVIDIA driver ≥ 535
- Docker
- ~50GB disk for model weights

### 1. Build the optimized image

```bash
# Clone this repo
git clone https://github.com/<your-username>/qwen38-27b-dual-3080.git
cd qwen38-27b-dual-3080

# Build (starts from local vllm/vllm-openai:v0.28.0 if available, else pulls it)
docker build -f docker/Dockerfile.syv-upg -t qwen38-27b:syv-upg .
```

### 2. Download model weights (on host, NOT in container)

```bash
# Create a clean env for downloading
mamba create -p ~/hfenv python=3.12 pip -y
source ~/hfenv/bin/activate
pip install huggingface_hub>=1.31 hf_transfer

# Download (host has IPv4 fallback; container doesn't)
HF_HUB_DISABLE_XET=1 HF_XET_ENABLED=0 no_proxy='*' \
  hf download Qwen/Qwen3.8-27B \
  --local-dir ~/models/Qwen3.8-27B-NVFP4
```

### 3. Launch

```bash
# Daily config: 224K context, MTP, TP=2
./scripts/run-serve.sh \
  CTX=fast MAX_LEN=229376 GPU_UTIL=0.86 MAX_SEQS=4 \
  SPEC=mtp DRAFT_TOKENS=4 MTP_DRAFT_VOCAB=0 PREFIX_CACHE=1 \
  TP=2 PORT=8000
```

Wait ~5 minutes (torch.compile + KV pool init). Then:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## Benchmark Results

Full data in [`bench/`](bench/). Summary from 9-23 A/B test (6 samples per config, median):

### Single-stream decode (C1, thinking off, 224K context available)

| Task | tok/s | Acceptance | Notes |
|---|---|---|---|
| Code generation | **100-103** | ~50% | 6-sample median, range 92-110 |
| Counting (1-1000) | **175** | ~93% | MTP thrives on repetitive output |
| Chinese writing | **81** | ~40% | 224K ctx, full quality |
| 12K context | **116** | — | decode stable up to 200K+ |
| 170K context | **84** | — | graceful decay (SSM helps) |
| 215K context | **72** | — | still interactive |

### Prefill

| Context | tok/s | TTFT |
|---|---|---|
| 5.7K | 752 | 7.6s |
| 22.5K | 1,261 | 17.9s |
| 51.5K | 1,276 | 40.4s |

### Prefix caching

| Scenario | First TTFT | Second TTFT | Speedup |
|---|---|---|---|
| 25,776 token doc | 22.1s | 1.7s | **13×** |

### Concurrency

| Streams | Per-stream | Aggregate |
|---|---|---|
| 1 | 103 | 103 |
| 4 | ~25 | ~100 |

MTP advantage vanishes at concurrency ≥4 (batch full, no idle verification capacity).

## The 43-Patch Stack

Our Docker image applies patches on top of vLLM 0.28.0:

| # | Patch | What it does |
|---|---|---|
| 1 | `lm_head-int8` | Requantize lm_head + embed_tokens to INT8 (saves 2.6GB) |
| 2 | `qwen35-embed-quant` | Route embedding to vLLM's quantized kernel |
| 3 | `mamba-fp16-state` | 16-bit recursive state (37→64 concurrent seats) |
| 4 | `int8-act` | INT4 weights × INT8 activations via INT8 tensor cores (4× rate) |
| 5 | `int8-negative-scales` | Fix AutoRound ~50% negative scale bug (critical!) |
| 6 | `draft-vocab-40k` | 40K draft vocab covering 97.5% of model output |
| 7 | `split-kv-triton` | Split-KV verification attention (57µs→23µs/layer) |
| 8 | `unsorted-sampler` | Skip full-vocab sort in top-k/top-p |
| 9 | `dflash2-backport` | DFlash2 block drafter (optional, TP=1 only) |
| 10 | `hybrid-prefix-cache` | Prefix caching for hybrid SSM+attention |
| 11-43 | ... | Bug fixes, robustness, serving improvements |

Full patch list: [`docker/patches/`](docker/patches/)

## 4 Critical Constraints (Learned the Hard Way)

1. **`MTP_DRAFT_VOCAB=0` is mandatory** — the default 40K draft vocab is Danish+English+Python. Without this flag, Chinese acceptance crashes to 2% (55→19 tok/s). With it, Chinese recovers to 81 tok/s.

2. **`GPU_UTIL=0.86`, not 0.90** — GPU0 is the display card with ~1.6GB desktop overhead. At 0.90, free memory drops to 0.34GB and the first torch.compile OOM-kills the engine (HTTP 000).

3. **`PREFIX_CACHE=1` must stay on** — long document round-2 TTFT drops from 287-386s to 2.5-5.3s.

4. **`--tensor-parallel-size 2`** must be in `EXTRA_ARGS` (the launcher only reads TP from EXTRA_ARGS).

## Why Not Just Use llama.cpp?

We tested llama.cpp extensively (see [`refs/llama.cpp-comparison.md`](refs/llama.cpp-comparison.md)):

| | llama.cpp (best: Q8_0 + MTP + tensor) | Our vLLM stack |
|---|---|---|
| 2× 3080 decode | ~45-55 tok/s (extrapolated) | **100-103 tok/s** |
| 4× 3080 decode | 95 tok/s (published, 220W) | **103 tok/s (2 cards!)** |
| KV cache | f16 required for tensor mode | FP8 (32KB/token) |
| Context fit | 256K tight in 20GB/card | 224K comfortable |
| Prefix cache | No | 13× TTFT reduction |
| Chinese | Acceptance drops ~40% | 81 tok/s (with draft vocab fix) |

The gap is the **INT8 tensor core path** + **split-KV Triton** — vLLM's kernel ecosystem has optimizations that llama.cpp doesn't match yet for this model architecture.

## Reproducing Our Numbers

```bash
# Clone
git clone https://github.com/<your-username>/qwen38-27b-dual-3080.git

# Build
docker build -f docker/Dockerfile.syv-upg -t qwen38-27b:syv-upg .

# Download model (on host)
mamba create -p ~/hfenv python=3.12 pip -y && source ~/hfenv/bin/activate
pip install huggingface_hub hf_transfer
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3.8-27B --local-dir ~/models/Qwen3.8-27B-NVFP4

# Launch
./scripts/run-serve.sh CTX=fast MAX_LEN=229376 GPU_UTIL=0.86 \
  SPEC=mtp DRAFT_TOKENS=4 MTP_DRAFT_VOCAB=0 PREFIX_CACHE=1 TP=2

# Benchmark
python3 bench/bench_decode.py --base http://127.0.0.1:8000
python3 bench/bench_context.py --base http://127.0.0.1:8000
```

## Comparison with Other Published Results

| Rig | Stack | Config | Decode | Source |
|---|---|---|---|---|
| **2× 3080 20G (ours)** | **vLLM 0.28 + 43p** | **MTP, TP=2, 320W** | **100-103** | **this repo** |
| 4× 3080 20G | llama.cpp | Q8_0, MTP, tensor, 220W | 95 | [jdkruzr/3080bench](https://github.com/jdkruzr/3080bench) |
| 3× 3090 24G | llama.cpp | MTP, tensor | 95.6 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) |
| 2× 5060 Ti 16G | llama.cpp | MTP, tensor, n-max4 | 71.3 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) |
| 1× 3090 24G | llama.cpp | MTP, n-max2 | 41.3 | [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) |
| 1× 3090 24G | vLLM + syv-ai | MTP, C1, 250W | 114-124 | [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) |

## License

Deployment scripts and benchmark code: MIT. Model weights: see Qwen/Qwen3.8-27B license.
