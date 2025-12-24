#!/bin/bash
#
# Overnight GPU Experiment Runner for Assignment 5
# Runs SFT, Expert Iteration, and GRPO experiments for 10 hours total
#
# Usage: ./run_overnight_experiments.sh
#

set -e  # Exit on error

# Configuration
TOTAL_HOURS=10
OUTPUT_BASE="overnight_results_$(date +%Y%m%d_%H%M%S)"
TRAIN_DATA="math_rlvr/chat-fmt-base.jsonl"
MODEL="Qwen/Qwen2.5-Math-1.5B"
BATCH_SIZE=1  # Reduced to 1 due to limited GPU memory (54GB already in use)

# Time budgets (in minutes)
SFT_TIME=$((TOTAL_HOURS * 60 / 3))           # ~200 minutes
EI_TIME=$((TOTAL_HOURS * 60 / 3))            # ~200 minutes
GRPO_TIME=$((TOTAL_HOURS * 60 / 3))          # ~200 minutes

HPARAM_SEARCH_TIME=60  # 60 minutes per method for hparam search
FINAL_TRAIN_TIME=$((SFT_TIME - HPARAM_SEARCH_TIME))  # ~140 minutes

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUTPUT_BASE/experiment.log"
}

log "========================================================================"
log "OVERNIGHT GPU EXPERIMENTS - ASSIGNMENT 5"
log "========================================================================"
log "Total time budget: $TOTAL_HOURS hours"
log "Time per method: $((TOTAL_HOURS / 3)) hours"
log "Output directory: $OUTPUT_BASE"
log "Training data: $TRAIN_DATA"
log "========================================================================"

# Activate virtual environment
source .venv/bin/activate

# ============================================================================
# PHASE 1: SUPERVISED FINE-TUNING (SFT)
# ============================================================================

log ""
log "========================================================================"
log "PHASE 1: SUPERVISED FINE-TUNING (SFT)"
log "========================================================================"

SFT_DIR="$OUTPUT_BASE/sft"
mkdir -p "$SFT_DIR"

# Hyperparameter search - test 4 configs (15 mins each)
log "Step 1: Hyperparameter search (4 configs x 15 mins = 60 mins)"

SEARCH_STEPS=400  # ~15 minutes at 2-3 sec/step

configs=(
    "1e-5:0.1:2:0.5"
    "2e-5:0.1:2:1.0"
    "5e-5:0.2:2:1.0"
    "3e-5:0.15:4:1.0"
)

best_loss=999999
best_config=""
best_run_dir=""

for i in "${!configs[@]}"; do
    IFS=':' read -r lr warmup grad_accum grad_norm <<< "${configs[$i]}"
    run_name="sft_search_$(printf '%02d' $((i+1)))"
    run_dir="$SFT_DIR/$run_name"
    mkdir -p "$run_dir"

    log "  Config $((i+1))/4: LR=$lr, warmup=$warmup, grad_accum=$grad_accum, grad_norm=$grad_norm"

    timeout 20m python sft_qwen_math.py \
        --train-data "$TRAIN_DATA" \
        --model-name "$MODEL" \
        --output-dir "$run_dir" \
        --max-steps $SEARCH_STEPS \
        --learning-rate $lr \
        --warmup-ratio $warmup \
        --gradient-accumulation-steps $grad_accum \
        --max-grad-norm $grad_norm \
        --batch-size $BATCH_SIZE \
        --save-steps 200 \
        --logging-steps 10 \
        --bf16 \
        2>&1 | tee "$run_dir/training.log" || log "  Config $((i+1)) failed or timed out"

    # Extract final loss
    if [ -f "$run_dir/metrics.jsonl" ]; then
        final_loss=$(tail -1 "$run_dir/metrics.jsonl" | python -c "import json, sys; print(json.load(sys.stdin)['loss'])" 2>/dev/null || echo "999999")
        log "  Final loss: $final_loss"

        # Track best config
        if (( $(echo "$final_loss < $best_loss" | bc -l) )); then
            best_loss=$final_loss
            best_config="${configs[$i]}"
            best_run_dir="$run_dir"
        fi
    fi
done

if [ -z "$best_config" ]; then
    log "WARNING: No successful search runs. Using default config."
    best_config="5e-5:0.2:2:1.0"
    best_loss="N/A"
fi

log "Best SFT config: $best_config (loss: $best_loss)"

# Train final model with best config
log ""
log "Step 2: Training final SFT model with best config (~140 mins)"

IFS=':' read -r lr warmup grad_accum grad_norm <<< "$best_config"
SFT_FINAL="$SFT_DIR/final_model"
mkdir -p "$SFT_FINAL"

FINAL_STEPS=$((FINAL_TRAIN_TIME * 60 / 3))  # ~2800 steps for 140 mins

timeout $((FINAL_TRAIN_TIME + 10))m python sft_qwen_math.py \
    --train-data "$TRAIN_DATA" \
    --model-name "$MODEL" \
    --output-dir "$SFT_FINAL" \
    --max-steps $FINAL_STEPS \
    --learning-rate $lr \
    --warmup-ratio $warmup \
    --gradient-accumulation-steps $grad_accum \
    --max-grad-norm $grad_norm \
    --batch-size $BATCH_SIZE \
    --save-steps 500 \
    --logging-steps 20 \
    --bf16 \
    2>&1 | tee "$SFT_FINAL/training.log"

log "SFT phase complete! Model saved to: $SFT_FINAL"

# ============================================================================
# PHASE 2: EXPERT ITERATION
# ============================================================================

log ""
log "========================================================================"
log "PHASE 2: EXPERT ITERATION (FILTER-CORRECT SFT)"
log "========================================================================"

EI_DIR="$OUTPUT_BASE/expert_iteration"
mkdir -p "$EI_DIR"

log "Expert Iteration Strategy: Train only on correct solutions using --filter-correct flag"

# Hyperparameter search with filter-correct
log "Step 1: Hyperparameter search with --filter-correct (4 configs x 15 mins)"

best_loss=999999
best_config=""

for i in "${!configs[@]}"; do
    IFS=':' read -r lr warmup grad_accum grad_norm <<< "${configs[$i]}"
    run_name="ei_search_$(printf '%02d' $((i+1)))"
    run_dir="$EI_DIR/$run_name"
    mkdir -p "$run_dir"

    log "  Config $((i+1))/4: LR=$lr, warmup=$warmup, grad_accum=$grad_accum, grad_norm=$grad_norm"

    timeout 20m python sft_qwen_math.py \
        --train-data "$TRAIN_DATA" \
        --model-name "$MODEL" \
        --output-dir "$run_dir" \
        --filter-correct \
        --max-steps $SEARCH_STEPS \
        --learning-rate $lr \
        --warmup-ratio $warmup \
        --gradient-accumulation-steps $grad_accum \
        --max-grad-norm $grad_norm \
        --batch-size $BATCH_SIZE \
        --save-steps 200 \
        --logging-steps 10 \
        --bf16 \
        2>&1 | tee "$run_dir/training.log" || log "  Config $((i+1)) failed or timed out"

    if [ -f "$run_dir/metrics.jsonl" ]; then
        final_loss=$(tail -1 "$run_dir/metrics.jsonl" | python -c "import json, sys; print(json.load(sys.stdin)['loss'])" 2>/dev/null || echo "999999")
        log "  Final loss: $final_loss"

        if (( $(echo "$final_loss < $best_loss" | bc -l) )); then
            best_loss=$final_loss
            best_config="${configs[$i]}"
        fi
    fi
done

if [ -z "$best_config" ]; then
    log "WARNING: No successful search runs. Using default config."
    best_config="5e-5:0.2:2:1.0"
    best_loss="N/A"
fi

log "Best Expert Iteration config: $best_config (loss: $best_loss)"

# Train final model
log ""
log "Step 2: Training final Expert Iteration model (~140 mins)"

IFS=':' read -r lr warmup grad_accum grad_norm <<< "$best_config"
EI_FINAL="$EI_DIR/final_model"
mkdir -p "$EI_FINAL"

timeout $((FINAL_TRAIN_TIME + 10))m python sft_qwen_math.py \
    --train-data "$TRAIN_DATA" \
    --model-name "$MODEL" \
    --output-dir "$EI_FINAL" \
    --filter-correct \
    --max-steps $FINAL_STEPS \
    --learning-rate $lr \
    --warmup-ratio $warmup \
    --gradient-accumulation-steps $grad_accum \
    --max-grad-norm $grad_norm \
    --batch-size $BATCH_SIZE \
    --save-steps 500 \
    --logging-steps 20 \
    --bf16 \
    2>&1 | tee "$EI_FINAL/training.log"

log "Expert Iteration phase complete! Model saved to: $EI_FINAL"

# ============================================================================
# PHASE 3: GRPO (or Advanced SFT variant)
# ============================================================================

log ""
log "========================================================================"
log "PHASE 3: ADVANCED SFT (GRPO infrastructure not yet available)"
log "========================================================================"

GRPO_DIR="$OUTPUT_BASE/grpo_sft"
mkdir -p "$GRPO_DIR"

log "Running conservative SFT with lower learning rate as GRPO placeholder"
log "Using --include-ground-truth to augment training data"

# Hyperparameter search for conservative SFT
log "Step 1: Hyperparameter search with ground truth augmentation"

conservative_configs=(
    "5e-6:0.1:4:0.5"
    "1e-5:0.1:4:0.5"
    "2e-5:0.15:4:1.0"
    "3e-5:0.2:4:1.0"
)

best_loss=999999
best_config=""

for i in "${!conservative_configs[@]}"; do
    IFS=':' read -r lr warmup grad_accum grad_norm <<< "${conservative_configs[$i]}"
    run_name="grpo_search_$(printf '%02d' $((i+1)))"
    run_dir="$GRPO_DIR/$run_name"
    mkdir -p "$run_dir"

    log "  Config $((i+1))/4: LR=$lr, warmup=$warmup, grad_accum=$grad_accum, grad_norm=$grad_norm"

    timeout 20m python sft_qwen_math.py \
        --train-data "$TRAIN_DATA" \
        --model-name "$MODEL" \
        --output-dir "$run_dir" \
        --include-ground-truth \
        --filter-correct \
        --max-steps $SEARCH_STEPS \
        --learning-rate $lr \
        --warmup-ratio $warmup \
        --gradient-accumulation-steps $grad_accum \
        --max-grad-norm $grad_norm \
        --batch-size $BATCH_SIZE \
        --save-steps 200 \
        --logging-steps 10 \
        --bf16 \
        2>&1 | tee "$run_dir/training.log" || log "  Config $((i+1)) failed or timed out"

    if [ -f "$run_dir/metrics.jsonl" ]; then
        final_loss=$(tail -1 "$run_dir/metrics.jsonl" | python -c "import json, sys; print(json.load(sys.stdin)['loss'])" 2>/dev/null || echo "999999")
        log "  Final loss: $final_loss"

        if (( $(echo "$final_loss < $best_loss" | bc -l) )); then
            best_loss=$final_loss
            best_config="${conservative_configs[$i]}"
        fi
    fi
done

if [ -z "$best_config" ]; then
    log "WARNING: No successful search runs. Using default config."
    best_config="2e-5:0.15:4:1.0"
    best_loss="N/A"
fi

log "Best conservative SFT config: $best_config (loss: $best_loss)"

# Train final model
log ""
log "Step 2: Training final conservative SFT model (~140 mins)"

IFS=':' read -r lr warmup grad_accum grad_norm <<< "$best_config"
GRPO_FINAL="$GRPO_DIR/final_model"
mkdir -p "$GRPO_FINAL"

timeout $((FINAL_TRAIN_TIME + 10))m python sft_qwen_math.py \
    --train-data "$TRAIN_DATA" \
    --model-name "$MODEL" \
    --output-dir "$GRPO_FINAL" \
    --include-ground-truth \
    --filter-correct \
    --max-steps $FINAL_STEPS \
    --learning-rate $lr \
    --warmup-ratio $warmup \
    --gradient-accumulation-steps $grad_accum \
    --max-grad-norm $grad_norm \
    --batch-size $BATCH_SIZE \
    --save-steps 500 \
    --logging-steps 20 \
    --bf16 \
    2>&1 | tee "$GRPO_FINAL/training.log"

log "Conservative SFT phase complete! Model saved to: $GRPO_FINAL"

# ============================================================================
# SUMMARY
# ============================================================================

log ""
log "========================================================================"
log "EXPERIMENT COMPLETE!"
log "========================================================================"
log "Results saved to: $OUTPUT_BASE"
log ""
log "Trained models:"
log "  1. SFT: $SFT_FINAL"
log "  2. Expert Iteration: $EI_FINAL"
log "  3. Advanced SFT: $GRPO_FINAL"
log ""
log "Next steps:"
log "  1. Evaluate models on benchmark: math_rlvr/benchmark_dataset.jsonl"
log "  2. Compare performance and select best model"
log "  3. Analyze training curves in metrics.jsonl files"
log "========================================================================"

# Save summary
cat > "$OUTPUT_BASE/summary.txt" << EOF
Overnight Experiment Summary
============================

Start time: $(date)
Total duration: $TOTAL_HOURS hours

Models trained:
1. SFT: $SFT_FINAL
2. Expert Iteration: $EI_FINAL
3. Advanced SFT (GRPO placeholder): $GRPO_FINAL

Configuration:
- Training data: $TRAIN_DATA
- Base model: $MODEL
- Steps per final model: $FINAL_STEPS

Hyperparameter search results saved in each method's directory.
Training logs and metrics available in each run directory.
EOF

log "Summary written to: $OUTPUT_BASE/summary.txt"
