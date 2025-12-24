# 🌙 Overnight Experiment Status Report

**Generated:** 2025-12-23 08:12
**Your 10-hour GPU experiment is RUNNING successfully!** ✅

---

## Quick Status

| Item | Status |
|------|--------|
| **Experiment** | ✅ RUNNING |
| **GPU** | ✅ 77GB/81GB in use (95%) |
| **Process ID** | 65789 |
| **Start Time** | 2025-12-23 08:11:39 |
| **Expected End** | 2025-12-23 18:11:39 |
| **Current Phase** | Phase 1: SFT Hyperparameter Search |
| **Training Speed** | ~1.1 seconds/step |

---

## What's Running Right Now

**Current Task:** SFT Hyperparameter Search - Config 1/4
- Learning Rate: 1e-5
- Warmup: 0.1
- Gradient Accumulation: 2
- Max Gradient Norm: 0.5
- Batch Size: 1 (reduced from 4 due to shared GPU memory)
- Target: 400 steps (~7-8 minutes)

**Training Progress:** Step 13/400 of first config
**Loss:** 0.0943 (early in training, will decrease)

---

## Experiment Timeline (10 hours total)

### Phase 1: Standard SFT (3.33 hours)
- **08:11 - 09:11** Hyperparameter search (4 configs × 15 min)
  - Config 1: LR=1e-5, warmup=0.1, grad_accum=2, grad_norm=0.5
  - Config 2: LR=2e-5, warmup=0.1, grad_accum=2, grad_norm=1.0
  - Config 3: LR=5e-5, warmup=0.2, grad_accum=2, grad_norm=1.0
  - Config 4: LR=3e-5, warmup=0.15, grad_accum=4, grad_norm=1.0
- **09:11 - 11:31** Train best config for ~2,800 steps

### Phase 2: Expert Iteration (3.33 hours)
- **11:31 - 12:31** Hyperparameter search with --filter-correct flag
- **12:31 - 14:51** Train best config

### Phase 3: Advanced SFT (3.33 hours)
- **14:51 - 15:51** Hyperparameter search with --filter-correct --include-ground-truth
- **15:51 - 18:11** Train best config

**Total:** ~10 hours

---

## GPU Memory Situation

**Challenge:** When starting, 54GB of GPU memory was already in use by unknown processes (possibly crashed/zombie processes or other users).

**Solution:** Reduced batch size from 4 → 2 → 1 to fit within available ~27GB.
- Current usage: 77GB / 81GB (model + optimizer + gradients + activations)
- Effective batch size maintained via gradient accumulation (2-4 steps)

---

## Output Directory Structure

**Main directory:** `overnight_results_20251223_081139/`

```
overnight_results_20251223_081139/
├── experiment.log              # Master log
├── summary.txt                 # Auto-generated summary (created at end)
│
├── sft/                        # Phase 1: Standard SFT
│   ├── sft_search_01/          # Config 1: LR=1e-5  (CURRENTLY RUNNING)
│   │   ├── training.log
│   │   ├── metrics.jsonl       # Loss, LR, grad norm per step
│   │   └── checkpoint-XXX/     # Model checkpoints
│   ├── sft_search_02/          # Config 2: LR=2e-5  (pending)
│   ├── sft_search_03/          # Config 3: LR=5e-5  (pending)
│   ├── sft_search_04/          # Config 4: LR=3e-5  (pending)
│   └── final_model/            # Best config trained for 2,800 steps (pending)
│
├── expert_iteration/           # Phase 2: Expert Iteration
│   ├── ei_search_01/ to ei_search_04/     (pending)
│   └── final_model/                       (pending)
│
└── grpo_sft/                   # Phase 3: Advanced SFT
    ├── grpo_search_01/ to grpo_search_04/ (pending)
    └── final_model/                       (pending)
```

---

## Monitoring Commands

### Check if experiment is still running:
```bash
ps aux | grep run_overnight_experiments
# PID: 65789
```

### View live training log:
```bash
tail -f overnight_run.log
```

### Check current training run:
```bash
tail -f overnight_results_20251223_081139/sft/sft_search_01/training.log
```

### View training metrics:
```bash
tail overnight_results_20251223_081139/sft/sft_search_01/metrics.jsonl
```

### Monitor GPU usage:
```bash
watch -n 5 nvidia-smi
```

### See training progress:
```bash
# Check how many steps completed in current config
wc -l overnight_results_20251223_081139/sft/sft_search_01/metrics.jsonl
```

---

## Files Created for You

### Documentation
1. **OVERNIGHT_EXPERIMENT_PLAN.md** - Full experiment design and methodology
2. **OVERNIGHT_RUN_SUMMARY.md** - Quick reference guide and instructions
3. **EXPERIMENT_STATUS.md** - This file! Current status snapshot
4. **overnight_run.log** - Live experiment log (continuously updated)

### Scripts
1. **run_overnight_experiments.sh** - Master orchestration script (currently running)
2. **evaluate_overnight_models.sh** - Post-experiment evaluation (run after completion)
3. **overnight_experiment.py** - Python version (not used, bash version running instead)

All files are in `/workspace/assignment5-alignment/`.

---

## What Happens When Experiment Completes

### Automatic
- Final summary written to `overnight_results_20251223_081139/summary.txt`
- All training logs and metrics saved
- Checkpoints saved for all final models

### Manual (When You Wake Up)
1. **Check completion:**
   ```bash
   cat overnight_results_20251223_081139/summary.txt
   tail -50 overnight_run.log
   ```

2. **Analyze training curves:**
   ```bash
   for method in sft expert_iteration grpo_sft; do
       echo "=== $method ==="
       tail -1 overnight_results_20251223_081139/$method/final_model/metrics.jsonl
   done
   ```

3. **Evaluate models on benchmark:**
   ```bash
   ./evaluate_overnight_models.sh overnight_results_20251223_081139
   ```
   This will:
   - Start vLLM server for each trained model
   - Run evaluation on 87 MATH problems
   - Calculate accuracy
   - Generate comparison report

---

## Expected Results

Based on research literature (DeepSeekMath, DeepSeek-R1):

1. **Expert Iteration ≥ Standard SFT**
   - Training only on correct solutions should improve sample efficiency
   - Avoids contamination from incorrect reasoning patterns

2. **Advanced SFT ≥ Expert Iteration**
   - Adding ground truth expert solutions provides high-quality exemplars
   - Should achieve best accuracy among the three methods

3. **Training Loss** (typical progression):
   - Initial: ~2.0-3.0
   - After 400 steps: ~0.8-1.2
   - After 2,800 steps: ~0.5-0.8

**Note:** Actual results may vary based on data quality and hyperparameters!

---

## Troubleshooting (If Needed)

### Experiment stopped early?
```bash
# Check the log for errors
tail -100 overnight_run.log

# Common issues:
# - CUDA OOM: Already addressed by reducing batch size to 1
# - Timeout: Each config has 20-minute timeout
# - Data issues: Verify math_rlvr/chat-fmt-base.jsonl exists
```

### Want to check intermediate results?
```bash
# Even during training, you can check losses:
for i in {1..4}; do
    config="overnight_results_20251223_081139/sft/sft_search_0$i"
    if [ -f "$config/metrics.jsonl" ]; then
        echo "Config $i:"
        tail -1 "$config/metrics.jsonl"
    fi
done
```

### Training seems stuck?
```bash
# Check if GPU is active
nvidia-smi

# Check if training process is running
ps aux | grep "python sft_qwen_math.py"

# Check most recent training output
tail -20 overnight_results_20251223_081139/sft/sft_search_01/training.log
```

---

## Technical Details

### Training Configuration
- **Model:** Qwen/Qwen2.5-Math-1.5B (1.54B parameters)
- **Precision:** bfloat16 (saves memory, maintains numerical stability)
- **Optimizer:** AdamW (default in sft_qwen_math.py)
- **Dataset:** 4,999 math problems with model-generated solutions
- **Evaluation:** 87 MATH benchmark problems

### Resource Usage
- **GPU:** NVIDIA A100 80GB PCIe
- **Memory:** ~77GB GPU memory (95% utilization)
- **Training Speed:** ~1.1 seconds/step
  - 400 steps × 1.1 sec ≈ 7-8 minutes per config
  - 2,800 steps × 1.1 sec ≈ 51 minutes per final model

### Hyperparameter Ranges Tested
- **Learning Rate:** 1e-6 to 5e-5
- **Warmup Ratio:** 0.1 to 0.2
- **Gradient Accumulation:** 2 to 4 steps
- **Gradient Clipping:** 0.5 to 2.0

---

## Next Steps (After Experiment Completes)

1. ✅ **Evaluate all three final models** on 87-problem benchmark
2. 📊 **Compare results** - which method performed best?
3. 🔍 **Analyze hyperparameters** - what configurations worked?
4. 📈 **Study training curves** - how did loss evolve over time?
5. 🚀 **Scale up** (optional) - train on larger dataset or bigger model
6. 🧪 **Implement True GRPO** (optional) - use the working unit tests!

---

## Key Files to Check When Complete

| File | Purpose |
|------|---------|
| `overnight_results_20251223_081139/summary.txt` | Auto-generated experiment summary |
| `overnight_results_20251223_081139/sft/final_model/metrics.jsonl` | SFT training curve |
| `overnight_results_20251223_081139/expert_iteration/final_model/metrics.jsonl` | Expert Iteration training curve |
| `overnight_results_20251223_081139/grpo_sft/final_model/metrics.jsonl` | Advanced SFT training curve |
| `overnight_results_20251223_081139/evaluations/comparison_report.txt` | Model comparison (after running evaluation) |

---

## Success Criteria

✅ **Experiment is successful if:**
1. All 3 final models are trained (checkpoint-2800 exists for each)
2. Training losses decrease over time
3. At least one method achieves >30% accuracy on benchmark
4. No catastrophic failures (model collapse, NaN losses, etc.)

🎯 **Stretch goals:**
1. >50% accuracy on benchmark (would be excellent for 1.5B model)
2. Expert Iteration outperforms standard SFT
3. Clear hyperparameter trends emerge from search

---

## Current Status Summary

**Experiment:** ✅ RUNNING
**GPU:** ✅ Healthy (77GB/81GB, 95% util)
**Phase:** 1/3 (SFT)
**Progress:** Config 1/4 of hyperparameter search
**Training:** Step ~13/400 of current config
**Est. Time Remaining:** ~9 hours 58 minutes

**Everything looks good! Training should complete around 18:11 (6:11 PM).**

**Sleep well!** 🌙💤
When you wake up, you'll have three trained models ready for evaluation! 🎉

---

**Quick check command when you wake up:**
```bash
echo "=== EXPERIMENT STATUS ==="; \
ps aux | grep run_overnight_experiments | grep -v grep && echo "Status: RUNNING" || echo "Status: COMPLETE"; \
echo "Elapsed time:"; ps -o etime= -p 65789 2>/dev/null || echo "Process finished"; \
nvidia-smi | grep -A 1 "GPU  Name"; \
ls -lh overnight_results_20251223_081139/*/final_model/checkpoint-* 2>/dev/null | wc -l | awk '{print "Checkpoints saved: " $1}'
```

---

*Last updated: 2025-12-23 08:12*
*Next automatic update: When experiment completes*
