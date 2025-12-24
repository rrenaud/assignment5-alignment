#!/bin/bash

# Launch a vLLM OpenAI-compatible server for Qwen2.5 Math 1.5B.
# Usage: ./launch_vllm.sh [additional vLLM arguments]

set -euo pipefail

MODEL_NAME="Qwen/Qwen2.5-Math-1.5B"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"

echo "Starting vLLM server for $MODEL_NAME"
echo " - Host: $HOST"
echo " - Port: $PORT"
echo " - Tensor parallel size: $TP_SIZE"
echo " - GPU memory utilization: $GPU_UTIL"

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "$@"
