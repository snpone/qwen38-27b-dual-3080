#!/bin/bash
# run-serve.sh — Start Qwen3.8-27B serving on dual GPU
#
# Usage:
#   ./run-serve.sh CTX=fast MAX_LEN=229376 GPU_UTIL=0.86 \
#                  SPEC=mtp DRAFT_TOKENS=4 MTP_DRAFT_VOCAB=0 PREFIX_CACHE=1 TP=2
#
# Environment variables (defaults in quotes):
#   CTX="fast"         Context profile: fast=229376, med=196608, full=262144
#   MAX_LEN=""         Override max model length (empty = auto from CTX)
#   GPU_UTIL="0.86"    GPU memory utilization per card
#   MAX_SEQS="4"       Max concurrent sequences
#   SPEC="mtp"         Speculative decoding: mtp | dflash2 | none
#   DRAFT_TOKENS="4"   MTP draft tokens per step
#   MTP_DRAFT_VOCAB="0" 0=full 248K vocab (Chinese-safe), 1=40K (English-only, faster)
#   PREFIX_CACHE="1"   1=on (recommended), 0=off
#   TP="2"             Tensor parallelism size
#   PORT="8000"        API port
#   MODEL=""           Model path (default: /host/models/Qwen3.8-27B-NVFP4)
#   MODEL_NAME="qwen3.8-27b"  Model name in API
#   CONTAINER="qwen38-27b"    Container name
#   IMG="qwen38-27b:syv-upg"  Docker image
#   CUDA_DEVS="0,1"    GPU devices
#   NPROC=""           nproc override (empty = auto)
#   PWR_LIMIT=""       nvidia-smi pl override (empty = stock 320W)
#   EXTRA_ARGS=""      Extra vLLM CLI args

set -euo pipefail

CTX="${CTX:-fast}"
MAX_LEN="${MAX_LEN:-}"
GPU_UTIL="${GPU_UTIL:-0.86}"
MAX_SEQS="${MAX_SEQS:-4}"
SPEC="${SPEC:-mtp}"
DRAFT_TOKENS="${DRAFT_TOKENS:-4}"
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-0}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/host/models/Qwen3.8-27B-NVFP4}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"
CONTAINER="${CONTAINER:-qwen38-27b}"
IMG="${IMG:-qwen38-27b:syv-upg}"
CUDA_DEVS="${CUDA_DEVS:-0,1}"
NPROC="${NPROC:-}"
PWR_LIMIT="${PWR_LIMIT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Auto-resolve max length from profile
if [ -z "$MAX_LEN" ]; then
  case "$CTX" in
    fast)  MAX_LEN=229376 ;;
    med)   MAX_LEN=196608 ;;
    full)  MAX_LEN=262144 ;;
    *)     MAX_LEN=$CTX ;;  # treat as number
  esac
fi

# Build vLLM args
VLLM_ARGS="--served-model-name $MODEL_NAME"
VLLM_ARGS="$VLLM_ARGS --max-model-len $MAX_LEN"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization $GPU_UTIL"
VLLM_ARGS="$VLLM_ARGS --max-num-seqs $MAX_SEQS"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8"
VLLM_ARGS="$VLLM_ARGS --max-num-batched-tokens 16384"
VLLM_ARGS="$VLLM_ARGS --disable-async-output-proc"
VLLM_ARGS="$VLLM_ARGS --reasoning-parser qwen3"
VLLM_ARGS="$VLLM_ARGS --trust-remote-code"
VLLM_ARGS="$VLLM_ARGS --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice"
VLLM_ARGS="$VLLM_ARGS --tool-call-parser qwen3_coder"
VLLM_ARGS="$VLLM_ARGS --distributed-executor-backend mp"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size $TP"

# Speculative decoding
if [ "$SPEC" = "mtp" ]; then
  VLLM_ARGS="$VLLM_ARGS --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_tensor_parallel_size\":$TP}'"
  # Env for MTP draft vocab
  MTP_ENV="MTP_DRAFT_VOCAB=$MTP_DRAFT_VOCAB"
  MTP_ENV="$MTP_ENV MTP_ENABLE_FULL_VOCAB=$MTP_DRAFT_VOCAB"
  MTP_ENV="$MTP_ENV MTP_FULL_VOCAB_CACHE_DIR=/mtp_full"
fi

if [ "$PREFIX_CACHE" = "1" ]; then
  VLLM_ARGS="$VLLM_ARGS --enable-prefix-caching"
fi

VLLM_ARGS="$VLLM_ARGS $EXTRA_ARGS"

# nproc
if [ -n "$NPROC" ]; then
  NPROC_ARG="-n $NPROC"
else
  NPROC_ARG="-n"
fi

echo "=== Starting $MODEL_NAME ==="
echo "  Container:  $CONTAINER"
echo "  Image:      $IMG"
echo "  Model:      $MODEL"
echo "  CTX:        $MAX_LEN tokens"
echo "  GPU util:   $GPU_UTIL per card"
echo "  Max seqs:   $MAX_SEQS"
echo "  TP:         $TP"
echo "  Spec:       $SPEC (draft=$DRAFT_TOKENS, draft_vocab=$MTP_DRAFT_VOCAB)"
echo "  Prefix:     $PREFIX_CACHE"
echo "  Port:       $PORT"
echo "  nproc:      ${NPROC:-auto}"
echo "  Power:      ${PWR_LIMIT:-stock}"
echo "  CUDA dev:   $CUDA_DEVS"
echo ""

# Kill existing
docker rm -f "$CONTAINER" 2>/dev/null || true

# Build env var string
ENV_ARGS=""
if [ "$SPEC" = "mtp" ]; then
  ENV_ARGS="-e $MTP_ENV"
fi

docker run -d --rm --name "$CONTAINER" \
  --gpus "\"device=$CUDA_DEVS\"" \
  --ipc=host \
  -p "$PORT:8000" \
  -v "$MODEL:$MODEL:ro" \
  $ENV_ARGS \
  $IMG \
  $NPROC_ARG bash -c "
    export VLLM_USE_V1=1
    export VLLM_TORCH_ATTENTION_USE_OUT_OF_PLACE_ALLOCATION=1
    export MAMBA_SSM_DTYPE=float16
    export TORCH_COMPILE_MAX_AUTOTUNE=0
    export VLLM_FP8_ATTENTION=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    exec vllm serve $MODEL $VLLM_ARGS
  "

echo ""
echo "Container started. Wait ~5 min for torch.compile."
echo "Check:  docker logs -f $CONTAINER"
echo "API:     http://127.0.0.1:$PORT/v1/chat/completions"
echo "Health:  curl http://127.0.0.1:$PORT/v1/models"
