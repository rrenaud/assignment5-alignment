# SFT Training Metrics Guide

This guide describes the comprehensive metrics tracking system implemented in `sft_qwen_math.py`.

## Overview

The training pipeline now includes:
- **TensorBoard integration** for real-time visualization
- **JSON logging** for detailed metric analysis
- **Per-token metrics** including special tracking for `\boxed{}` tokens
- **Loss breakdown** by subject and difficulty level
- **Gradient monitoring** to detect training instabilities

## Metrics Tracked

### Core Training Metrics

| Metric | Description | Logged Every |
|--------|-------------|--------------|
| `loss/train` | Overall training loss (averaged) | logging_steps (default: 10) |
| `learning_rate` | Current learning rate from scheduler | logging_steps |
| `gradient_norm` | L2 norm of gradients (pre-clipping) | logging_steps |
| `epoch` | Current epoch number | Every step |

### Detailed Token-Level Metrics

Computed every `detailed_metrics_steps` (default: 100 steps):

| Metric | Description |
|--------|-------------|
| `loss/total` | Total loss for the batch |
| `loss/perplexity` | exp(loss) - measure of prediction uncertainty |
| `loss/per_token_avg` | Average loss per valid token |
| `loss/boxed_tokens` | **Loss specifically on tokens within `\boxed{}`** |
| `loss/non_boxed_tokens` | Loss on all other tokens |
| `accuracy/token` | Token-level accuracy (% correct predictions) |
| `accuracy/boxed_tokens` | **Accuracy on `\boxed{}` tokens specifically** |
| `stats/total_tokens` | Number of valid (non-masked) tokens |
| `stats/boxed_token_count` | Number of tokens inside `\boxed{}` |
| `stats/batch_size` | Batch size for this step |

### Epoch-Level Metrics

Logged at the end of each epoch:

| Metric | Description |
|--------|-------------|
| `epoch_loss/total` | Average loss for entire epoch |
| `epoch_loss/subject/{subject}` | Loss breakdown by math subject |
| `epoch_loss/level/{level}` | Loss breakdown by difficulty level |

**Subjects tracked:**
- Algebra
- Counting and Probability
- Geometry
- Intermediate Algebra
- Number Theory
- Prealgebra
- Precalculus

**Levels tracked:** Level 1 through Level 5

## Output Files

### 1. JSON Metrics Log
**Location:** `{output_dir}/metrics_{timestamp}.jsonl`

Each line is a JSON object with timestamped metrics:
```json
{
  "loss/train": 2.3456,
  "learning_rate": 1.5e-05,
  "gradient_norm": 0.85,
  "loss/boxed_tokens": 2.1234,
  "accuracy/token": 0.456,
  "step": 100,
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

### 2. TensorBoard Logs
**Location:** `{output_dir}/tensorboard/`

View with: `tensorboard --logdir {output_dir}/tensorboard`

## Why Track `\boxed{}` Tokens Separately?

The `\boxed{answer}` format contains the **final answer** to math problems. Tracking loss/accuracy specifically on these tokens helps you:

1. **Monitor answer quality** - Are we learning to produce correct final answers?
2. **Detect overfitting** - High accuracy on reasoning but low on boxed answers suggests memorization
3. **Prioritize optimization** - You could weight boxed token loss higher in future versions
4. **Debug failures** - Compare boxed vs non-boxed loss to see where the model struggles

## Command-Line Options

```bash
# Basic logging (every 10 steps)
python3 sft_qwen_math.py --logging-steps 10

# Detailed metrics every 50 steps (default: 100)
python3 sft_qwen_math.py --detailed-metrics-steps 50

# Disable TensorBoard (only JSON logging)
python3 sft_qwen_math.py --disable-tensorboard
```

## Additional Metrics Worth Monitoring

Here are suggestions for additional metrics you could implement:

### 1. **Generation-Based Metrics** (Recommended)
- **Validation pass during training**: Generate answers on a held-out validation set
- **Exact match accuracy**: % of problems where generated answer matches ground truth
- **\boxed{} format compliance**: % of generations that include proper `\boxed{}` formatting
- **Solution length distribution**: Track avg length of generated solutions vs ground truth

**Implementation:** Add a validation loop every N steps that generates answers and compares to ground truth.

### 2. **Attention Metrics**
- **Attention entropy**: Measure focus/diffusion of attention patterns
- **Cross-layer attention consistency**: Track how attention patterns evolve through layers
- **Key token attention**: Attention weights on mathematical operators, numbers, `\boxed{}`

**Use case:** Debugging whether model attends to relevant mathematical concepts.

### 3. **Optimizer Health Metrics**
- **Per-layer gradient norms**: Detect vanishing/exploding gradients in specific layers
- **Parameter update ratios**: `||update|| / ||params||` per layer
- **Learning rate warmup visualization**: Track LR schedule over time

**Use case:** Diagnose training instabilities, tune hyperparameters.

### 4. **Token-Type Specific Losses**
Expand beyond `\boxed{}` to track:
- **Mathematical operators**: `+, -, \times, \frac, \sqrt`
- **Numbers vs text**: Track loss on numeric tokens separately
- **LaTeX formatting**: Track loss on LaTeX commands (`\begin`, `\end`, `\displaystyle`)

**Use case:** Identify if model struggles with specific mathematical notation.

### 5. **Diversity Metrics**
- **Vocabulary usage**: % of vocabulary actively used
- **N-gram diversity**: Unique n-grams in generated text
- **Template detection**: Track if model is repeating phrases

**Use case:** Detect mode collapse or overfitting to specific solution patterns.

### 6. **Comparison to Ground Truth**
- **Per-level accuracy**: Track exact match % by Level 1-5
- **Per-subject accuracy**: Track exact match % by subject
- **Error type classification**: Categorize errors (computational, conceptual, formatting)

**Use case:** Identify which problem types need more training data.

### 7. **Model Confidence Metrics**
- **Prediction entropy**: Measure model's uncertainty token-by-token
- **Top-k probability mass**: % of probability on top-k predictions
- **Calibration**: Compare predicted probabilities to actual accuracy

**Use case:** Understand when the model is confident vs uncertain.

### 8. **Computational Metrics**
- **GPU memory usage**: Track VRAM consumption
- **Throughput**: Tokens/second, samples/second
- **Time per step**: Breakdown of forward/backward/optimizer time

**Use case:** Optimize training efficiency.

## Analyzing Metrics

### Quick Analysis with Python

```python
import json
import pandas as pd

# Load metrics
metrics = []
with open('output_dir/metrics_*.jsonl') as f:
    for line in f:
        metrics.append(json.loads(line))

df = pd.DataFrame(metrics)

# Plot loss over time
import matplotlib.pyplot as plt
plt.plot(df['step'], df['loss/train'], label='Total Loss')
plt.plot(df['step'], df['loss/boxed_tokens'], label='Boxed Token Loss')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

# Compare boxed vs non-boxed loss
print(f"Avg Boxed Loss: {df['loss/boxed_tokens'].mean():.4f}")
print(f"Avg Non-Boxed Loss: {df['loss/non_boxed_tokens'].mean():.4f}")
```

### Using TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./qwen_math_sft_output/tensorboard --port 6006

# Open http://localhost:6006 in your browser
```

**Useful views:**
- **Scalars**: Line plots of all metrics over time
- **Distributions**: Gradient norm distributions
- **Compare runs**: Overlay multiple training runs

## Best Practices

1. **Monitor gradient norms**: Should stay roughly stable (0.5-2.0 range). Sudden spikes indicate instability.

2. **Track boxed loss separately**: If boxed loss >> total loss, model struggles with final answers.

3. **Compare subjects/levels**: Identify weak areas to oversample in training data.

4. **Validate regularly**: Generate actual answers every N steps to catch issues early.

5. **Log hyperparameters**: Save all args to a JSON file for reproducibility.

6. **Use TensorBoard smoothing**: Raw metrics are noisy; smooth them for trend analysis.

## Troubleshooting

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Gradient norm >> 10 | Exploding gradients | Lower learning rate, increase gradient clipping |
| Loss plateaus early | Learning rate too low | Increase LR or reduce warmup |
| Boxed loss much higher than total | Model can't learn answer format | Add more \boxed{} examples, increase weight on boxed tokens |
| High variance in loss | Batch size too small | Increase batch size or gradient accumulation |
| Perplexity > 100 | Model not learning | Check data quality, increase model capacity |

## Example Training Commands

```bash
# Full monitoring with frequent detailed metrics
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./output_full_monitoring \
    --logging-steps 5 \
    --detailed-metrics-steps 25 \
    --save-steps 500 \
    --bf16

# Minimal overhead (only basic metrics)
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./output_minimal \
    --logging-steps 50 \
    --detailed-metrics-steps 500 \
    --disable-tensorboard \
    --bf16
```

## Summary

The enhanced metrics system provides:
- ✅ Real-time loss tracking with TensorBoard
- ✅ Per-token accuracy monitoring
- ✅ **Special tracking for `\boxed{}` answer tokens**
- ✅ Loss breakdown by subject and difficulty
- ✅ Gradient health monitoring
- ✅ JSON logs for custom analysis

This gives you comprehensive visibility into training dynamics and helps identify issues early.
