# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS336 Spring 2025 Assignment 5: Alignment - A machine learning course assignment focused on alignment techniques including Supervised Fine-Tuning (SFT), Expert Iteration, and Group Relative Policy Optimization (GRPO) with verified rewards on the MATH dataset.

This repository includes both the main assignment and an optional supplemental assignment on safety alignment, instruction tuning, and RLHF.

## Development Commands

### Setup and Installation

```bash
# Install dependencies (flash-attn requires special handling)
uv sync --no-install-package flash-attn
uv sync
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_sft.py
uv run pytest tests/test_grpo.py
uv run pytest tests/test_dpo.py

# Run specific test function
uv run pytest tests/test_sft.py::test_tokenize_prompt_and_output -v
```

### Submission

```bash
# Run tests and create submission zip
./test_and_make_submission.sh
```

## Architecture Overview

### Core Implementation Pattern

The codebase uses an **adapter pattern** for student implementations. Students implement functions in `tests/adapters.py`, which are then called by test files:

- **`tests/adapters.py`**: Contains stubs prefixed with `run_*` that students implement
- **Test files** (`test_sft.py`, `test_grpo.py`, `test_dpo.py`): Import and test the adapter functions
- **Snapshot testing**: Tests use `numpy_snapshot` fixtures to compare outputs against expected results in `tests/_snapshots/`

### Main Assignment Components

**Supervised Fine-Tuning (SFT)**:
- `run_tokenize_prompt_and_output`: Tokenizes prompts/outputs and creates response masks
- `run_sft_microbatch_train_step`: Computes SFT loss and performs backpropagation
- `run_masked_normalize`: Sums over dimensions and normalizes by a constant

**Group Relative Policy Optimization (GRPO)**:
- `run_compute_group_normalized_rewards`: Normalizes rewards within groups (for GRPO)
- `run_get_response_log_probs`: Computes log probabilities and optional token entropy
- `run_compute_naive_policy_gradient_loss`: Basic REINFORCE loss
- `run_compute_grpo_clip_loss`: PPO-style clipped policy gradient
- `run_compute_policy_gradient_loss`: Wrapper that delegates to specific loss types
- `run_grpo_microbatch_train_step`: Computes policy gradient loss and backprops

**Shared Utilities**:
- `run_masked_mean`: Computes mean over masked elements
- `run_compute_entropy`: Calculates entropy from logits

### Optional Assignment Components (Safety/RLHF)

Located in `tests/adapters.py` (bottom section):
- `get_packed_sft_dataset`: Creates packed instruction-tuning datasets
- `run_iterate_batches`: Batch iterator for datasets
- `run_parse_mmlu_response`: Parses MMLU multiple-choice responses
- `run_parse_gsm8k_response`: Extracts numeric answers from GSM8K outputs
- `run_compute_per_instance_dpo_loss`: Computes DPO (Direct Preference Optimization) loss

### Math Grading System

The `cs336_alignment/drgrpo_grader.py` module provides high-recall math answer verification:
- Based on HuggingFace math_verify, verl, and other verification systems
- Normalizes LaTeX expressions and compares them symbolically
- Handles various math formats (fractions, intervals, matrices, etc.)

### Auxiliary Scripts

**`math_rlvr/` directory** contains optional experimentation code:
- `sft_qwen_math.py`: Custom SFT training script for Qwen-2.5-Math-1.5B
- `eval_math.py`, `query.py`: Evaluation and inference utilities
- `param_sweep.py`, `test_sweep.py`: Hyperparameter tuning
- `create_benchmark.py`: Benchmark creation tools

## Key Implementation Details

### Gradient Accumulation

The training step functions take a `gradient_accumulation_steps` parameter. Losses should be normalized by this value before calling `.backward()` to ensure correct gradient scaling.

### Response Masking

For SFT, only the assistant's response tokens (not the prompt) should contribute to the loss:
- `response_mask`: 1 for response tokens, 0 for prompt/padding tokens
- Use `masked_mean` or `masked_normalize` to respect this mask

### GRPO Loss Types

Three policy gradient variants are supported:
- `"no_baseline"`: Uses raw rewards directly
- `"reinforce_with_baseline"`: Uses group-normalized advantages
- `"grpo_clip"`: PPO-style clipping with advantages and old log probs

### Normalization Constants

Some loss functions support optional `normalize_constant` parameters:
- For SFT: Typically set to 1.0 (default) or the sequence length
- For GRPO: Can normalize by different constants for variance reduction

## Data Structure

**Training Data** (`data/` directory):
- `alpaca_eval/`: AlpacaEval prompts and annotations
- `gsm8k/`: Grade school math problems
- `mmlu/`: Massive Multitask Language Understanding benchmark
- `simple_safety_tests/`: Safety evaluation prompts

**Test Fixtures** (`tests/fixtures/`):
- `tiny-gpt2/`: Minimal GPT-2 model for testing
- `Meta-Llama-3-8B/`: Tokenizer for Llama 3
- `tokenized_sft_sample.json`: Sample tokenized data

## Testing Philosophy

- Tests compare against **snapshot files** in `tests/_snapshots/`
- If snapshots don't exist or need updating, pytest will create/update them
- Snapshots ensure implementation matches expected behavior exactly
- To regenerate snapshots: `uv run pytest --snapshot-update`

## Common Gotchas

1. **flash-attn installation**: Must be installed separately with `--no-install-package` first
2. **Adapter functions**: All student implementations go in `tests/adapters.py`, not scattered across files
3. **Tensor shapes**: Pay attention to batch dimensions - some operations expect (batch, seq_len), others expect scalar rewards of shape (batch, 1)
4. **Loss scaling**: Don't forget to divide by `gradient_accumulation_steps` before `.backward()`
5. **Mask handling**: Always check whether a mask is boolean or float (0/1), and whether masked elements should be zeroed or ignored


## Maintain claude_log.txt

Any time you run user written scripts/commands, add the command to claude_log.txt .  

