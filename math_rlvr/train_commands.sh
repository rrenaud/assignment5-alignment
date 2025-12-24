#!/bin/bash
# Example training commands for SFT on Qwen-2.5-Math-1.5B

# Basic training (default settings)
# - 3 epochs
# - batch size 4, gradient accumulation 4 (effective batch size 16)
# - learning rate 2e-5
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_output \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --bf16

# Training with only correct responses (if responses are verified)
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_correct_only \
    --filter-correct \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --bf16

# Training with ground truth solutions included
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_with_gt \
    --include-ground-truth \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --bf16

# Resume from checkpoint
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_output \
    --resume-from ./qwen_math_sft_output/checkpoint-epoch1-step1000 \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --bf16

# Larger batch training (if you have more GPU memory)
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_large_batch \
    --num-epochs 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5 \
    --bf16

# Lower learning rate for more stable training
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_stable \
    --num-epochs 5 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.05 \
    --bf16
