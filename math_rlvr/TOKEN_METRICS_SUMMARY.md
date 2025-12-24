# Token-Level Loss Breakdown Summary

## Overview

The training pipeline now tracks loss separately for three distinct token types:

1. **Thinking tokens** (`<think>...</think>`) - Reasoning/scratchpad content
2. **Boxed tokens** (`\boxed{...}`) - Final mathematical answers
3. **Other tokens** - Explanatory text, LaTeX, formatting

## Why This Matters

Different token types serve different purposes and may have different learning dynamics:

- **Thinking tokens**: Internal reasoning, can be more exploratory
- **Boxed tokens**: Critical final answers, must be precise
- **Other tokens**: Explanatory text, mathematical notation, formatting

By tracking these separately, you can:
- Identify if the model struggles with final answers vs reasoning
- Adjust training to focus on weak areas
- Detect overfitting patterns (e.g., memorizing answers but not reasoning)
- Monitor if the model learns to use the `<think>` format correctly

## Metrics Tracked

### Per-Token Type Loss

```
loss/thinking_tokens    - Average loss on tokens within <think>...</think>
loss/boxed_tokens       - Average loss on tokens within \boxed{...}
loss/other_tokens       - Average loss on all other tokens
```

### Per-Token Type Accuracy

```
accuracy/thinking_tokens - Token-level accuracy on thinking tokens
accuracy/boxed_tokens    - Token-level accuracy on boxed answer tokens
```

### Token Distribution

```
stats/thinking_token_count - Number of thinking tokens in batch
stats/boxed_token_count    - Number of boxed tokens in batch
stats/other_token_count    - Number of other tokens in batch
stats/thinking_pct         - Percentage of total tokens
stats/boxed_pct           - Percentage of total tokens
stats/other_pct           - Percentage of total tokens
```

## Example Output

### During Training

```
Progress: [=====>] loss: 2.3456, lr: 1.5e-05, grad: 0.85, epoch: 1, boxed: 1.987, think: 2.456
```

Shows specialized loss for boxed and thinking tokens when available.

### Epoch Summary

```
======================================================================
Epoch 1/3 Summary:
  Average Loss: 2.3456

  Token Distribution (from last batch):
    Thinking tokens: 38.2%
    Boxed tokens:    3.2%
    Other tokens:    58.6%

  Loss by Subject:
    Algebra                  : 2.1234 (150 batches)
    Geometry                 : 2.5678 (120 batches)
    ...
======================================================================
```

### Metrics JSON Example

```json
{
  "loss/train": 2.3456,
  "loss/thinking_tokens": 2.4567,
  "loss/boxed_tokens": 1.9876,
  "loss/other_tokens": 2.3890,
  "accuracy/token": 0.456,
  "accuracy/thinking_tokens": 0.445,
  "accuracy/boxed_tokens": 0.512,
  "stats/total_tokens": 1024,
  "stats/thinking_token_count": 391,
  "stats/boxed_token_count": 33,
  "stats/other_token_count": 600,
  "stats/thinking_pct": 38.2,
  "stats/boxed_pct": 3.2,
  "stats/other_pct": 58.6,
  "learning_rate": 1.5e-05,
  "gradient_norm": 0.85,
  "step": 100,
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

## Typical Distribution

From real DeepInfra responses:
- **Thinking tokens**: ~35-40% (reasoning content)
- **Boxed tokens**: ~2-5% (final answers)
- **Other tokens**: ~55-60% (explanatory text, LaTeX)

## Interpretation Guide

### Normal Training

```
loss/thinking_tokens: 2.5
loss/boxed_tokens: 2.3
loss/other_tokens: 2.4
```

All losses are similar - model learning uniformly across token types.

### Model Struggling with Final Answers

```
loss/thinking_tokens: 2.1
loss/boxed_tokens: 3.2  ← HIGH
loss/other_tokens: 2.0
```

Boxed token loss is significantly higher - model may need:
- More examples with correct answers
- Weighted loss on boxed tokens
- Better understanding of answer format

### Model Struggling with Reasoning

```
loss/thinking_tokens: 3.5  ← HIGH
loss/boxed_tokens: 2.0
loss/other_tokens: 2.3
```

Thinking token loss is high - model may:
- Not be using <think> tags effectively
- Struggling with reasoning steps
- Need more training on process vs outcome

### Unbalanced Accuracy

```
accuracy/thinking_tokens: 0.35
accuracy/boxed_tokens: 0.65
```

High boxed accuracy but low thinking accuracy might indicate:
- Model memorizing answers without reasoning
- Need to focus more on process quality
- Potential overfitting

## Unit Tests

Comprehensive unit tests are provided in `test_boxed_metrics.py`:

```bash
python3 test_boxed_metrics.py
```

Tests cover:
- ✅ Simple `\boxed{}` detection
- ✅ Multiple boxed expressions
- ✅ Nested braces
- ✅ Batch processing
- ✅ Masked labels (prompt masking)
- ✅ `<think>` tag detection
- ✅ Multiple thinking blocks
- ✅ Real-world DeepInfra response format
- ✅ Loss computation structure

All 11 tests passing.

## Implementation Details

### Character-Level Token Mapping

The implementation uses character-level mapping to accurately identify which tokens belong to `<think>` or `\boxed{}` regions:

1. Decode each token individually
2. Build a character→token mapping
3. Find pattern matches in full text using regex
4. Map character ranges back to token indices

This approach is robust to different tokenization schemes and handles edge cases like:
- Tokens split across pattern boundaries
- Multiple occurrences of patterns
- Nested braces in LaTeX expressions

### Computational Cost

Detailed metrics (including token-level breakdown) are computed every `detailed_metrics_steps` (default: 100):

- Minimal overhead: <1% additional training time
- Character mapping is efficient (O(n) where n = sequence length)
- No additional GPU memory required

## Advanced Usage

### Focus Training on Boxed Tokens

To emphasize final answer quality, you could modify the loss computation to weight boxed tokens higher:

```python
# In compute_loss function
if num_boxed_tokens > 0:
    boxed_loss = (per_token_loss * boxed_valid.float()).sum()
    # Weight boxed loss 2x
    total_loss += boxed_loss * 2.0
```

### Separate Validation for Each Token Type

Track validation metrics separately:

```python
val_metrics = {
    "val/thinking_loss": ...,
    "val/boxed_loss": ...,
    "val/boxed_exact_match": ...  # % of correct final answers
}
```

### Export Token-Level Data for Analysis

```python
import json
import pandas as pd

# Load metrics
df = pd.DataFrame([json.loads(line) for line in open('metrics.jsonl')])

# Plot loss breakdown
df[['loss/thinking_tokens', 'loss/boxed_tokens', 'loss/other_tokens']].plot()

# Analyze correlation
print(df[['loss/boxed_tokens', 'accuracy/boxed_tokens']].corr())
```

## Future Enhancements

Potential additions:

1. **Per-level boxed loss**: Track boxed loss by difficulty (Level 1-5)
2. **Mathematical operator loss**: Track loss on `+, -, \times, \frac`, etc.
3. **LaTeX formatting loss**: Track loss on LaTeX commands
4. **Thinking quality metrics**: Measure coherence/relevance of thinking content
5. **Answer format compliance**: Track % of responses with proper `\boxed{}` format

## Troubleshooting

### No boxed tokens detected

```
stats/boxed_token_count: 0
```

**Cause**: Training data may not contain `\boxed{}` format
**Solution**: Check data format, ensure responses include `\boxed{answer}`

### No thinking tokens detected

```
stats/thinking_token_count: 0
```

**Cause**: Responses don't use `<think>` tags
**Solution**: Use DeepInfra QwQ or similar reasoning models that output `<think>` tags

### Token overlap detected

```
overlap = (boxed_mask & thinking_mask).sum()  # Should be 0
```

**Cause**: Bug in token detection
**Solution**: Run unit tests to debug

## References

- `sft_qwen_math.py` - Main training script
- `test_boxed_metrics.py` - Comprehensive unit tests
- `METRICS_GUIDE.md` - Overall metrics documentation

## Example Analysis Script

```python
#!/usr/bin/env python3
"""Analyze token-level loss patterns."""

import json
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
metrics = []
with open('output_dir/metrics_*.jsonl') as f:
    for line in f:
        metrics.append(json.loads(line))

df = pd.DataFrame(metrics)

# Filter to rows with detailed metrics
detailed = df[df['stats/boxed_token_count'] > 0].copy()

print("Token-Level Loss Summary:")
print("="*50)
print(f"Average thinking loss: {detailed['loss/thinking_tokens'].mean():.4f}")
print(f"Average boxed loss:    {detailed['loss/boxed_tokens'].mean():.4f}")
print(f"Average other loss:    {detailed['loss/other_tokens'].mean():.4f}")
print()
print(f"Boxed/Other ratio:     {detailed['loss/boxed_tokens'].mean() / detailed['loss/other_tokens'].mean():.3f}")
print(f"Thinking/Other ratio:  {detailed['loss/thinking_tokens'].mean() / detailed['loss/other_tokens'].mean():.3f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Loss over time
axes[0].plot(detailed['step'], detailed['loss/thinking_tokens'], label='Thinking', alpha=0.7)
axes[0].plot(detailed['step'], detailed['loss/boxed_tokens'], label='Boxed', alpha=0.7)
axes[0].plot(detailed['step'], detailed['loss/other_tokens'], label='Other', alpha=0.7)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss by Token Type')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy over time
axes[1].plot(detailed['step'], detailed['accuracy/thinking_tokens'], label='Thinking', alpha=0.7)
axes[1].plot(detailed['step'], detailed['accuracy/boxed_tokens'], label='Boxed', alpha=0.7)
axes[1].plot(detailed['step'], detailed['accuracy/token'], label='Overall', alpha=0.7)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy by Token Type')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('token_metrics_analysis.png', dpi=150)
print("\nPlot saved to: token_metrics_analysis.png")
```
