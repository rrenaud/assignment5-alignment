#!/bin/bash
#
# Post-Experiment Evaluation Script
# Evaluates all models from overnight experiment on math benchmark
#
# Prerequisites:
# 1. vLLM must be installed
# 2. Overnight experiment must be complete
#
# Usage: ./evaluate_overnight_models.sh <results_directory>
#

set -e

if [ -z "$1" ]; then
    # Find most recent overnight results directory
    RESULTS_DIR=$(ls -dt overnight_results_* 2>/dev/null | head -1)
    if [ -z "$RESULTS_DIR" ]; then
        echo "Error: No overnight results directory found"
        echo "Usage: $0 <results_directory>"
        exit 1
    fi
    echo "Using most recent results directory: $RESULTS_DIR"
else
    RESULTS_DIR="$1"
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory $RESULTS_DIR does not exist"
    exit 1
fi

# Configuration
EVAL_DATA="math_rlvr/benchmark_dataset.jsonl"
EVAL_SAMPLES=87
VLLM_PORT=8000
VLLM_HOST="localhost"

# Create evaluation output directory
EVAL_DIR="$RESULTS_DIR/evaluations"
mkdir -p "$EVAL_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$EVAL_DIR/evaluation.log"
}

log "========================================================================"
log "POST-EXPERIMENT EVALUATION"
log "========================================================================"
log "Results directory: $RESULTS_DIR"
log "Evaluation data: $EVAL_DATA"
log "Number of problems: $EVAL_SAMPLES"
log "========================================================================"

# Find all final models
MODELS=(
    "$RESULTS_DIR/sft/final_model"
    "$RESULTS_DIR/expert_iteration/final_model"
    "$RESULTS_DIR/grpo_sft/final_model"
)

METHOD_NAMES=(
    "sft"
    "expert_iteration"
    "advanced_sft"
)

# Function to evaluate a model
evaluate_model() {
    local model_path="$1"
    local method_name="$2"
    local output_file="$EVAL_DIR/${method_name}_eval.jsonl"

    log ""
    log "Evaluating $method_name model: $model_path"

    # Find latest checkpoint
    local checkpoint=$(ls -d "$model_path"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$checkpoint" ]; then
        log "WARNING: No checkpoint found for $method_name, trying model directory directly"
        checkpoint="$model_path"
    fi

    if [ ! -d "$checkpoint" ]; then
        log "ERROR: Checkpoint directory not found: $checkpoint"
        return 1
    fi

    log "Using checkpoint: $checkpoint"

    # Check if model files exist
    if [ ! -f "$checkpoint/config.json" ] && [ ! -f "$checkpoint/pytorch_model.bin" ] && [ ! -d "$checkpoint/model.safetensors" ]; then
        log "ERROR: Model files not found in checkpoint directory"
        return 1
    fi

    # Start vLLM server
    log "Starting vLLM server on port $VLLM_PORT..."

    vllm serve "$checkpoint" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        > "$EVAL_DIR/${method_name}_vllm.log" 2>&1 &

    local vllm_pid=$!
    log "vLLM server started with PID: $vllm_pid"

    # Wait for server to be ready
    log "Waiting for vLLM server to be ready..."
    local retries=0
    local max_retries=60
    while ! curl -s "http://$VLLM_HOST:$VLLM_PORT/v1/models" > /dev/null 2>&1; do
        sleep 2
        retries=$((retries + 1))
        if [ $retries -ge $max_retries ]; then
            log "ERROR: vLLM server failed to start after ${max_retries} attempts"
            kill $vllm_pid 2>/dev/null || true
            return 1
        fi
    done
    log "vLLM server is ready!"

    # Run evaluation
    log "Running evaluation..."
    cd math_rlvr

    python eval_math.py \
        --split test \
        --num-samples "$EVAL_SAMPLES" \
        --jsonl-output "../$output_file" \
        --prompt-style chat \
        --workers 4 \
        2>&1 | tee "../$EVAL_DIR/${method_name}_eval_output.log"

    cd ..

    # Stop vLLM server
    log "Stopping vLLM server..."
    kill $vllm_pid 2>/dev/null || true
    wait $vllm_pid 2>/dev/null || true
    sleep 5

    # Calculate accuracy
    if [ -f "$output_file" ]; then
        local accuracy=$(python3 -c "
import json
with open('$output_file') as f:
    results = [json.loads(line) for line in f]
correct = sum(1 for r in results if r.get('correct', False))
total = len(results)
print(f'{correct}/{total} ({correct/total*100:.1f}%)')
")
        log "Accuracy: $accuracy"
    else
        log "ERROR: Evaluation output file not found"
    fi

    log "Evaluation complete for $method_name"
}

# Evaluate each model
for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    method_name="${METHOD_NAMES[$i]}"

    if [ -d "$model_path" ]; then
        evaluate_model "$model_path" "$method_name"
    else
        log "WARNING: Model directory not found: $model_path (skipping)"
    fi
done

# Generate comparison report
log ""
log "========================================================================"
log "GENERATING COMPARISON REPORT"
log "========================================================================"

python3 << 'EOF' > "$EVAL_DIR/comparison_report.txt"
import json
import os
from pathlib import Path

results_dir = Path(os.environ['EVAL_DIR'])
methods = ['sft', 'expert_iteration', 'advanced_sft']
method_labels = {
    'sft': 'Standard SFT',
    'expert_iteration': 'Expert Iteration',
    'advanced_sft': 'Advanced SFT'
}

print("="*80)
print("OVERNIGHT EXPERIMENT RESULTS COMPARISON")
print("="*80)
print()

results = {}
for method in methods:
    eval_file = results_dir / f"{method}_eval.jsonl"
    if eval_file.exists():
        with open(eval_file) as f:
            data = [json.loads(line) for line in f]
        correct = sum(1 for r in data if r.get('correct', False))
        total = len(data)
        accuracy = correct / total * 100 if total > 0 else 0
        results[method] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }

if not results:
    print("No evaluation results found!")
else:
    # Sort by accuracy
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    print("ACCURACY RANKINGS:")
    print("-" * 80)
    for rank, (method, data) in enumerate(sorted_methods, 1):
        label = method_labels.get(method, method)
        print(f"{rank}. {label:25s} {data['correct']:2d}/{data['total']:2d} ({data['accuracy']:5.1f}%)")

    print()
    print("DETAILED RESULTS:")
    print("-" * 80)

    for method in methods:
        if method in results:
            label = method_labels.get(method, method)
            data = results[method]
            print(f"\n{label}:")
            print(f"  Accuracy: {data['accuracy']:.1f}% ({data['correct']}/{data['total']})")

            # Try to get training metrics
            model_dir = Path(os.environ['RESULTS_DIR']) / method.replace('_sft', '_sft').replace('expert_iteration', 'expert_iteration') / 'final_model'
            metrics_file = model_dir / 'metrics.jsonl'

            if metrics_file.exists():
                with open(metrics_file) as f:
                    lines = f.readlines()
                    if lines:
                        first_metrics = json.loads(lines[0])
                        last_metrics = json.loads(lines[-1])
                        print(f"  Training steps: {last_metrics.get('step', 'N/A')}")
                        print(f"  Initial loss: {first_metrics.get('loss', 'N/A'):.4f}")
                        print(f"  Final loss: {last_metrics.get('loss', 'N/A'):.4f}")

    print()
    print("="*80)
    print("CONCLUSIONS:")
    print("="*80)

    if len(results) >= 2:
        best_method, best_data = sorted_methods[0]
        best_label = method_labels.get(best_method, best_method)
        print(f"\n• Best performing method: {best_label} ({best_data['accuracy']:.1f}%)")

        if 'sft' in results and 'expert_iteration' in results:
            improvement = results['expert_iteration']['accuracy'] - results['sft']['accuracy']
            if improvement > 0:
                print(f"• Expert Iteration outperformed SFT by {improvement:.1f} percentage points")
            else:
                print(f"• SFT outperformed Expert Iteration by {-improvement:.1f} percentage points")

    print()
EOF

cat "$EVAL_DIR/comparison_report.txt"

log "Comparison report saved to: $EVAL_DIR/comparison_report.txt"
log ""
log "========================================================================"
log "EVALUATION COMPLETE!"
log "========================================================================"
log "All results saved to: $EVAL_DIR"
log ""
log "Files generated:"
log "  - evaluation.log: Master evaluation log"
log "  - <method>_eval.jsonl: Per-problem results for each method"
log "  - <method>_eval_output.log: Evaluation stdout for each method"
log "  - <method>_vllm.log: vLLM server logs for each method"
log "  - comparison_report.txt: Summary comparison across all methods"
log "========================================================================"
