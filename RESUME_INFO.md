# Resume Info for SFT Training

## Command to Run
```bash
uv run python sft_qwen_math.py \
  --train-data /workspace/assignment5-alignment/math_rlvr/deepinfra_qwen3_32b_results.jsonl \
  --output-dir ./sft_output_test \
  --model-name Qwen/Qwen2.5-Math-1.5B \
  --max-steps 10 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --bf16 \
  --logging-steps 1 \
  --save-steps 100
```

## Context
- Script: `/workspace/assignment5-alignment/sft_qwen_math.py`
- Training data: `/workspace/assignment5-alignment/math_rlvr/deepinfra_qwen3_32b_results.jsonl`
- Model: Qwen/Qwen2.5-Math-1.5B
- Task: Run the SFT training script on small input and fix any problems

## Data Format Expected
The script expects JSONL with fields:
- `problem`: the math problem text
- `response`: the model's response
- Optional: `correct`, `ground_truth`
