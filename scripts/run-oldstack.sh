#!/bin/bash
# run-oldstack.sh — Start Qwen3.8-27B with the OLD (pre-HyperQwen) stack
#
# This is the 9-05 MTP 192k config — kept for A/B comparison.
# The new HyperQwen stack (43 patches) is ~68% faster on decode.
#
# Usage:
#   ./run-oldstack.sh [CTX=196608] [PORT=8000]

set -euo pipefail

CTX="${1:-196608}"
PORT="${2:-8000}"
CONTAINER="qwen38-27b-old"
MODEL="/host/models/Qwen3.8-27B-NVFP4"
MODEL_NAME="qwen3.8-27b"

# Build vLLM args (old stack: no int8 act, no split-kv, no unsorted sampler)
VLLM_ARGS="--served-model-name $MODEL_NAME"
VLLM_ARGS="$VLLM_ARGS --max-model-len $CTX"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization 0.86"
VLLM_ARGS="$VLLM_ARGS --max-num-seqs 4"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8"
VLLM_ARGS="$VLLM_ARGS --max-num-batched-tokens 16384"
VLLM_ARGS="$VLLM_ARGS --disable-async-output-proc"
VLLM_ARGS="$VLLM_ARGS --reasoning-parser qwen3"
VLLM_ARGS="$VLLM_ARGS --trust-remote-code"
VLLM_ARGS="$VLLM_ARGS --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice"
VLLM_ARGS="$VLLM_ARGS --tool-call-parser qwen3_coder"
VLLM_ARGS="$VLLM_ARGS --distributed-executor-backend mp"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size 2"
VLLM_ARGS="$VLLM_ARGS --enable-prefix-caching"
VLLM_ARGS="$VLLM_ARGS --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2,\"draft_tensor_parallel_size\":2}'"

echo "=== Starting $MODEL_NAME (OLD stack) ==="
echo "  CTX: $CTX, TP=2, MTP draft=2, port=$PORT"
echo ""

docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d --rm --name "$CONTAINER" \
  --gpus "\"device=0,1\"" \
  --ipc=host \
  -p "$PORT:8000" \
  -v "$MODEL:$MODEL:ro" \
  -e VLLM_USE_V1=1 \
  -e VLLM_TORCH_ATTENTION_USE_OUT_OF_PLACE_ALLOCATION=1 \
  -e MAMBA_SSM_DTYPE=float16 \
  -e TORCH_COMPILE_MAX_AUTOTUNE=0 \
  -e VLLM_FP8_ATTENTION=0 \
  -e MTP_DRAFT_VOCAB=0 \
  -e MTP_ENABLE_FULL_VOCAB=0 \
  qwen38-27b:old \
  -n bash -c "
    exec vllm serve $MODEL $VLLM_ARGS
  "

echo "Old stack started on port $PORT"
