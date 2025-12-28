#!/bin/bash

# Launch a vLLM OpenAI-compatible server for Qwen2.5 Math 1.5B or a local model.
# Usage: ./launch_vllm.sh [model_path] [additional vLLM arguments]
#
# If the first argument is a path to a local directory, it will be used as the model.
# Otherwise, defaults to the HuggingFace model Qwen/Qwen2.5-Math-1.5B.
#
# Examples:
#   ./launch_vllm.sh                           # Use default HF model
#   ./launch_vllm.sh /path/to/local/model      # Use local model
#   ./launch_vllm.sh Qwen/Qwen2.5-Math-7B      # Use different HF model

set -euo pipefail

DEFAULT_MODEL="Qwen/Qwen2.5-Math-1.5B"

# Check if first argument is a model path (local directory or HF model name)
if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
    MODEL_NAME="$1"
    shift
else
    MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
fi

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
