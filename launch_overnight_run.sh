#!/bin/bash
# Wait for hparam search to complete, then launch overnight training with best params

SEARCH_DIR="/workspace/assignment5-alignment/hparam_search_20251227_020755"
TRAIN_DATA="/workspace/assignment5-alignment/math_rlvr/deepinfra_qwen3_32b_results.jsonl"

echo "Waiting for hparam search to complete..."

# Wait for search to finish (no more sft_qwen processes from the search)
while pgrep -f "hparam_search_sft.py" > /dev/null; do
    sleep 60
    echo "$(date): Still running..."
done

echo "Hparam search completed!"

# Get best config
BEST_CONFIG=$(python3 -c "
import json
with open('${SEARCH_DIR}/results.json') as f:
    results = json.load(f)
valid = [r for r in results if r.get('val_loss') is not None]
best = min(valid, key=lambda x: x['val_loss'])
print(json.dumps(best))
")

echo "Best config: $BEST_CONFIG"

# Extract params
LR=$(echo "$BEST_CONFIG" | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['learning_rate'])")
BS=$(echo "$BEST_CONFIG" | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['batch_size'])")
GA=$(echo "$BEST_CONFIG" | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['gradient_accumulation'])")
WD=$(echo "$BEST_CONFIG" | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['weight_decay'])")
WR=$(echo "$BEST_CONFIG" | python3 -c "import json,sys; print(json.load(sys.stdin)['config']['warmup_ratio'])")
BEST_VAL=$(echo "$BEST_CONFIG" | python3 -c "import json,sys; print(json.load(sys.stdin)['val_loss'])")

echo ""
echo "Best hyperparameters (val_loss=$BEST_VAL):"
echo "  Learning rate: $LR"
echo "  Batch size: $BS"
echo "  Gradient accumulation: $GA"
echo "  Weight decay: $WD"
echo "  Warmup ratio: $WR"
echo ""

# Launch overnight run (50x steps = 100,000)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/workspace/assignment5-alignment/overnight_sft_${TIMESTAMP}"

echo "Launching overnight training run..."
echo "Output dir: $OUTPUT_DIR"
echo "Steps: 100000"

python sft_qwen_math.py \
    --train-data "$TRAIN_DATA" \
    --model-name "Qwen/Qwen2.5-Math-1.5B" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 100000 \
    --eval-steps 1000 \
    --batch-size "$BS" \
    --gradient-accumulation-steps "$GA" \
    --learning-rate "$LR" \
    --weight-decay "$WD" \
    --warmup-ratio "$WR" \
    --max-length 2048 \
    --logging-steps 100 \
    --save-steps 10000 \
    --bf16 \
    --num-workers 4 \
    --use-wandb \
    --wandb-project "rlvr" \
    --wandb-name "sft_overnight_${TIMESTAMP}"

echo "Overnight run completed!"
