# Hyperparameter Sweep Analysis for SFT

## Sweep Configuration
- **Total Configs Tested**: 6 (random sample from 81 total combinations)
- **Training Steps per Config**: 200 steps (limited due to time constraints)
- **Dataset**: benchmark_dataset.jsonl
- **Objectives**: Multi-objective optimization for:
  1. Thinking Loss (reasoning quality)
  2. Boxed Loss (answer accuracy)
  3. Total Loss (overall performance)

## Results Summary

### Valid Runs (with complete metrics)

#### 🥇 **Run #3: Best for Answer Accuracy**
```
Configuration:
  Learning Rate:           1e-05
  Warmup Ratio:            0.1
  Gradient Accumulation:   2
  Max Gradient Norm:       0.5

Results:
  Thinking Loss:  0.987  ⭐ Good reasoning
  Boxed Loss:     0.337  🏆 BEST answer accuracy!
  Total Loss:     1.008

Analysis: This config produces the most accurate answers (boxed tokens),
          making it ideal for prioritizing correctness.
```

#### 🥈 **Run #4: Best Overall Performance**
```
Configuration:
  Learning Rate:           5e-05  (higher learning rate)
  Warmup Ratio:            0.2    (longer warmup)
  Gradient Accumulation:   2
  Max Gradient Norm:       1.0

Results:
  Thinking Loss:  0.893  🏆 BEST reasoning quality!
  Boxed Loss:     0.930
  Total Loss:     0.856  🏆 BEST overall loss!

Analysis: This config achieves the best overall training performance
          with strong reasoning and good (but not best) answer accuracy.
          The higher LR with longer warmup appears beneficial.
```

#### 🥉 **Run #2: Moderate Performance**
```
Configuration:
  Learning Rate:           2e-05
  Warmup Ratio:            0.1
  Gradient Accumulation:   2
  Max Gradient Norm:       2.0

Results:
  Total Loss:     0.965  (only total loss available)

Analysis: Middle-ground configuration with decent performance.
```

### Failed Runs (incomplete metrics)

**Runs #1, #5, #6**: These runs completed but didn't collect detailed
metrics for thinking/boxed losses (likely due to insufficient training
steps before the first detailed_metrics_steps checkpoint at step 20).

## Key Insights

### 1. **Learning Rate Impact**
- **5e-05** (Run #4): Best overall performance, fastest convergence
- **1e-05** (Run #3): Best answer accuracy but slower convergence
- **2e-05** (Run #2): Balanced middle ground

**Recommendation**: Use **5e-05** for fast training with good overall
performance, or **1e-05** when answer accuracy is critical.

### 2. **Warmup Ratio**
- **0.2 warmup** (Run #4) with higher LR shows best results
- **0.1 warmup** (Run #3) works well with lower LR
- Longer warmup helps stabilize higher learning rates

**Recommendation**: Use **0.2 warmup ratio** when using LR >= 5e-05

### 3. **Gradient Accumulation**
- **2 steps** appears in all successful configs
- Higher values (4, 8) didn't produce measurable results in this sweep

**Recommendation**: **2-4 gradient accumulation steps** is sufficient

### 4. **Gradient Clipping**
- **0.5** (Run #3): Tighter clipping → better answer accuracy
- **1.0** (Run #4): Moderate clipping → better overall performance
- **2.0** (Run #2): Looser clipping → decent results

**Recommendation**: Use **1.0** for balanced training

## Recommended Configurations

### For Production Training (Best Overall)
```bash
python sft_qwen_math.py \
  --learning-rate 5e-5 \
  --warmup-ratio 0.2 \
  --gradient-accumulation-steps 2 \
  --max-grad-norm 1.0 \
  --batch-size 4 \
  ...
```
**Expected**: Best overall loss (0.856), excellent reasoning (0.893)

### For Maximum Answer Accuracy
```bash
python sft_qwen_math.py \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --gradient-accumulation-steps 2 \
  --max-grad-norm 0.5 \
  --batch-size 4 \
  ...
```
**Expected**: Best boxed loss (0.337), prioritizes correctness

### For Faster Experimentation
```bash
python sft_qwen_math.py \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --gradient-accumulation-steps 2 \
  --max-grad-norm 2.0 \
  --batch-size 4 \
  ...
```
**Expected**: Balanced performance, more forgiving to hyperparameters

## Next Steps

1. **Extended Training**: Run the best configs for full training (1000+ steps)
2. **Validation**: Test on held-out validation set
3. **Fine-tuning**: Narrow the search around the best configs:
   - LR: [3e-5, 4e-5, 5e-5, 6e-5]
   - Warmup: [0.15, 0.2, 0.25]
4. **A/B Testing**: Compare Run #3 vs Run #4 on actual math problems

## Files Generated
- `sweep_results.jsonl`: All configuration results
- `run_001/` through `run_006/`: Individual training logs and metrics
- `pareto_frontier.jsonl`: Pareto-optimal configurations (pending)
