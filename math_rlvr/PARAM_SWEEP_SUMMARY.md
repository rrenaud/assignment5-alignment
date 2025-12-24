# Parameter Sweep Summary

## Framework Overview

Multi-objective parameter sweep optimizing for 3 objectives:
1. **Thinking Loss** - Loss on `<think>...</think>` tokens
2. **Boxed Loss** - Loss on `\boxed{...}` answer tokens
3. **Total Loss** - Overall training loss

## Benchmark Dataset

- **Size**: 87 samples (stratified across levels and subjects)
- **Training time**: ~1 minute per epoch
- **Estimated sweep time**: ~2-3 minutes per configuration

**Distribution**:
- By Level: 15-22 samples per level (Level 1-5)
- By Subject: 10-21 samples per subject (7 math subjects)

## Parameter Grid

```python
{
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "warmup_ratio": [0.05, 0.1, 0.2],
    "gradient_accumulation_steps": [2, 4, 8],
    "max_grad_norm": [0.5, 1.0, 2.0],
}
```

**Total configurations**: 3 × 3 × 3 × 3 = 81

## Fixed Parameters

```python
{
    "batch_size": 4,
    "max_steps": 200,  # Fast benchmark
    "logging_steps": 10,
    "detailed_metrics_steps": 20,
}
```

## Multi-Objective Optimization

Uses **Pareto frontier** analysis:
- A configuration is Pareto-optimal if no other configuration is better in ALL objectives
- Identifies trade-offs between objectives
- Finds specialized configurations (e.g., best for thinking vs best for boxed loss)

## Running the Sweep

### Quick Test (2 configs)
```bash
python3 test_sweep.py
```

### Full Sweep (81 configs, ~2-3 hours)
```bash
python3 param_sweep.py \
    --sweep-dir ./full_sweep_results \
    --benchmark-data benchmark_dataset.jsonl
```

### Subset Sweep (12 configs, ~20 minutes)
```bash
python3 param_sweep.py \
    --sweep-dir ./quick_sweep_results \
    --benchmark-data benchmark_dataset.jsonl \
    --max-configs 12
```

## Analyzing Results

```bash
python3 visualize_sweep.py --sweep-dir ./full_sweep_results
```

**Outputs**:
- Statistical analysis
- Parameter impact analysis
- Pareto frontier identification
- 3D visualization (if matplotlib available)
- Parameter impact plots

## Expected Outcomes

### Example Trade-offs

**Best for Thinking Loss** might have:
- Lower learning rate
- Higher warmup
- Focus on reasoning quality

**Best for Boxed Loss** might have:
- Higher learning rate
- Lower warmup
- Focus on answer precision

**Best for Total Loss** might be:
- Balanced configuration
- Moderate learning rate
- Good overall performance

### Pareto Frontier Analysis

The Pareto frontier reveals:
1. **No-brainer wins**: Configurations better in all objectives
2. **Trade-off points**: Can't improve one without hurting another
3. **Specialized configs**: Optimized for specific objectives

## Files Generated

```
sweep_results/
├── sweep_results.jsonl          # All configurations and metrics
├── pareto_frontier.jsonl        # Pareto-optimal configurations
├── summary.txt                  # Text summary
├── pareto_frontier_3d.png       # 3D visualization
├── parameter_impact.png         # Parameter impact plots
└── run_XXX/                     # Individual run outputs
    ├── metrics_*.jsonl
    ├── training.log
    └── checkpoints/
```

## Interpreting Results

### Correlation Analysis

- **High correlation** between thinking and total loss → Total dominated by thinking
- **Low correlation** between thinking and boxed → Independent optimization needed
- **Negative correlation** → Fundamental trade-off exists

### Parameter Impact

- **Learning rate**: Typically most impactful
- **Warmup ratio**: Affects stability
- **Grad accum**: Affects effective batch size
- **Grad norm**: Prevents instabilities

## Test Run Results

From initial 2-config test:

| Config | LR | Warmup | GradAccum | Thinking | Boxed | Total |
|--------|-----|--------|-----------|----------|--------|-------|
| 1 | 2e-5 | 0.1 | 4 | 1.0229 | 0.4282 | 1.2219 |
| 2 | 5e-5 | 0.1 | 2 | 0.7161 | 0.2395 | 0.7555 |

**Observation**: Higher LR (5e-5) better on all objectives for this small sample!

## Next Steps

1. ✓ Create benchmark dataset
2. ✓ Implement sweep framework
3. ✓ Validate with test runs
4. **Run full sweep**
5. Analyze Pareto frontier
6. Select optimal configuration(s)
7. Train full model with best config

## Usage Tips

### For Quick Iteration
- Use small `max_configs` (12-24)
- Focus on learning rate and warmup
- Fix grad_accum=4, max_grad_norm=1.0

### For Thorough Search
- Run full 81 configs
- Analyze all parameter interactions
- Identify Pareto frontier
- Test top 3-5 configs on validation set

### For Production
- Select 1-2 Pareto-optimal configs
- Validate on held-out set
- Run full training (all data, multiple epochs)
- Monitor all three objectives

## Citation

Framework based on multi-objective optimization principles:
- Pareto efficiency
- Trade-off analysis
- Specialized vs generalist configurations
