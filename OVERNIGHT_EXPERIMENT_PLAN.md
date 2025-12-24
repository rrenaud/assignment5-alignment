# Overnight GPU Experiment Plan
## Assignment 5: Alignment Methods Comparison

**Start Time:** 2025-12-23 08:05:42
**Duration:** 10 hours
**GPU:** NVIDIA A100 80GB PCIe
**Training Data:** math_rlvr/chat-fmt-base.jsonl (4,999 examples)
**Eval Data:** math_rlvr/benchmark_dataset.jsonl (87 examples)

---

## Experiment Structure

### Time Allocation
- **Total:** 10 hours (600 minutes)
- **Per Method:** 3.33 hours (200 minutes)
  - Hyperparameter Search: 60 minutes (4 configs × 15 min each)
  - Final Model Training: 140 minutes

### Three Methods Being Compared

1. **Standard SFT (Supervised Fine-Tuning)**
   - Train on all examples (correct and incorrect)
   - Goal: Learn to imitate the response style

2. **Expert Iteration** (Implemented via `--filter-correct`)
   - Train only on verified correct solutions
   - Goal: Learn from successful problem-solving patterns only

3. **Advanced SFT** (Implemented via `--filter-correct --include-ground-truth`)
   - Train on correct solutions + ground truth solutions
   - Goal: Combine model-generated correct solutions with expert solutions

Note: True GRPO requires a more complex training loop with:
- Multiple rollouts per prompt
- Reward computation using math_verify
- Group-wise reward normalization
- PPO-style policy gradient updates

This is not yet implemented, so Phase 3 uses an advanced SFT variant as a placeholder.

---

## Hyperparameter Search Configurations

### Phase 1: Standard SFT
```
Config 1: LR=1e-5,  warmup=0.1,  grad_accum=2, grad_norm=0.5
Config 2: LR=2e-5,  warmup=0.1,  grad_accum=2, grad_norm=1.0
Config 3: LR=5e-5,  warmup=0.2,  grad_accum=2, grad_norm=1.0
Config 4: LR=3e-5,  warmup=0.15, grad_accum=4, grad_norm=1.0
```

### Phase 2: Expert Iteration (with --filter-correct)
```
Same configs as Phase 1, but training only on correct examples
```

### Phase 3: Advanced SFT (with --filter-correct --include-ground-truth)
```
Config 1: LR=5e-6,  warmup=0.1,  grad_accum=4, grad_norm=0.5
Config 2: LR=1e-5,  warmup=0.1,  grad_accum=4, grad_norm=0.5
Config 3: LR=2e-5,  warmup=0.15, grad_accum=4, grad_norm=1.0
Config 4: LR=3e-5,  warmup=0.2,  grad_accum=4, grad_norm=1.0
```
(More conservative learning rates since training on higher quality data)

---

## Training Parameters

**Fixed Across All Runs:**
- Model: Qwen/Qwen2.5-Math-1.5B
- Batch size: 4
- Precision: bf16
- Optimizer: AdamW (default in sft_qwen_math.py)

**Search Phase:**
- Steps: 400 (~15 minutes per config)
- Save interval: every 200 steps
- Logging: every 10 steps

**Final Training Phase:**
- Steps: ~2,800 (140 minutes at ~3 sec/step)
- Save interval: every 500 steps
- Logging: every 20 steps

---

## Output Structure

```
overnight_results_20251223_080542/
├── sft/
│   ├── sft_search_01/ to sft_search_04/    # Hyperparameter search runs
│   └── final_model/                         # Best config, full training
├── expert_iteration/
│   ├── ei_search_01/ to ei_search_04/
│   └── final_model/
├── grpo_sft/
│   ├── grpo_search_01/ to grpo_search_04/
│   └── final_model/
├── experiment.log                           # Master log file
└── summary.txt                              # Final summary

Each run directory contains:
- checkpoint-XXXX/        # Model checkpoints
- metrics.jsonl           # Training metrics (loss, step, time)
- training.log            # Training output
- tokenizer files         # Copied from base model
```

---

## Expected Timeline

| Time  | Elapsed | Phase | Activity |
|-------|---------|-------|----------|
| 08:05 | 0:00 | SFT | Start SFT hyperparameter search |
| 08:20 | 0:15 | SFT | Config 1 complete |
| 08:35 | 0:30 | SFT | Config 2 complete |
| 08:50 | 0:45 | SFT | Config 3 complete |
| 09:05 | 1:00 | SFT | Config 4 complete, select best |
| 11:25 | 3:20 | SFT → EI | SFT final training complete, start Expert Iteration search |
| 12:25 | 4:20 | EI | Expert Iteration search complete |
| 14:45 | 6:40 | EI → Adv | Expert Iteration final training complete, start Advanced SFT search |
| 15:45 | 7:40 | Adv | Advanced SFT search complete |
| 18:05 | 10:00 | Done | All training complete |

---

## Success Metrics

### During Training (Automatic)
- Training loss convergence
- Loss per component (thinking tokens, boxed tokens, total)
- Gradient norms
- Learning rate schedule

### Post-Training (Manual Evaluation Required)
1. **Accuracy on Benchmark** (math_rlvr/benchmark_dataset.jsonl)
   - Requires: vLLM server + eval_math.py
   - Metric: % of problems solved correctly
   - Gold standard: verified using math_verify library

2. **Training Efficiency**
   - Final training loss
   - Steps to convergence
   - GPU utilization

3. **Method Comparison**
   - SFT vs Expert Iteration vs Advanced SFT
   - Which method produces most accurate models?
   - Which hyperparameters work best for each method?

---

## Monitoring Commands

Check overall progress:
```bash
tail -f overnight_run.log
```

Check specific training run:
```bash
tail -f overnight_results_20251223_080542/sft/sft_search_01/training.log
```

View training metrics:
```bash
tail overnight_results_20251223_080542/sft/sft_search_01/metrics.jsonl
```

Check GPU usage:
```bash
watch -n 5 nvidia-smi
```

Get experiment status:
```bash
ps aux | grep run_overnight_experiments
ls -lh overnight_results_20251223_080542/*/final_model/checkpoint-*
```

---

## Post-Experiment Evaluation Plan

### 1. Extract Training Results
```bash
# For each method, get final loss
for method in sft expert_iteration grpo_sft; do
    echo "=== $method ==="
    for run in overnight_results_*/$ method/*/metrics.jsonl; do
        tail -1 "$run"
    done
done
```

### 2. Evaluate on Benchmark (Requires vLLM)
```bash
# For each final model:
# 1. Start vLLM server with model
# 2. Run eval_math.py with --jsonl-output to save results
# 3. Calculate accuracy

# Example for SFT model:
MODEL_PATH="overnight_results_20251223_080542/sft/final_model/checkpoint-2800"

# Start vLLM (in separate terminal or background)
vllm serve $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# Run evaluation
cd math_rlvr
python eval_math.py \
    --split test \
    --num-samples 87 \
    --jsonl-output sft_final_eval.jsonl \
    --prompt-style chat

# Repeat for expert_iteration and grpo_sft models
```

### 3. Compare Results
```bash
# Analyze accuracy by method
python -c "
import json
methods = ['sft', 'expert_iteration', 'grpo_sft']
for method in methods:
    with open(f'{method}_final_eval.jsonl') as f:
        results = [json.loads(line) for line in f]
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        print(f'{method}: {correct}/{total} = {correct/total*100:.1f}%')
"
```

---

## Key Questions to Answer

1. **Does filtering for correct solutions improve performance?**
   - Compare SFT vs Expert Iteration accuracies

2. **Does adding ground truth solutions help?**
   - Compare Expert Iteration vs Advanced SFT

3. **What are the optimal hyperparameters for each method?**
   - Analyze search results within each phase

4. **How does training data quality affect sample efficiency?**
   - Compare loss convergence rates across methods

5. **What's the best model overall?**
   - Highest benchmark accuracy wins!

---

## Notes and Observations

### Limitations of Current Setup

1. **No True GRPO Implementation**
   - Would require: rollout generation, reward verification, group normalization, PPO updates
   - Using advanced SFT as placeholder for now

2. **Expert Iteration is Simplified**
   - True EI would involve: train model → generate solutions → verify → retrain on verified
   - Current implementation just filters existing correct solutions from training data

3. **Limited Evaluation**
   - 87 benchmark problems is small
   - May want to evaluate on full MATH test set (5,000 problems) if time permits

### Potential Improvements

1. **Implement Full Expert Iteration Loop**
   - Generate solutions with trained model
   - Verify correctness using math_verify
   - Create augmented dataset with verified solutions
   - Retrain on combined dataset

2. **Implement True GRPO**
   - Build rollout generation pipeline
   - Implement group-wise reward normalization (already tested in unit tests!)
   - Add PPO-style policy gradient training loop
   - Compare against SFT baselines

3. **More Sophisticated Hyperparameter Search**
   - Use validation set to select configs
   - Try more configurations
   - Use Bayesian optimization or other search strategies

---

## Expected Outcomes

Based on prior research (DeepSeekMath, etc.):

1. **Expert Iteration should outperform standard SFT**
   - Training only on correct solutions should be more sample-efficient
   - Model learns good reasoning patterns, not errors

2. **Ground truth solutions should provide additional benefit**
   - High-quality human solutions complement model-generated ones
   - May achieve best accuracy

3. **True GRPO would likely perform best**
   - RL from verified rewards is powerful
   - Group normalization reduces variance
   - But requires more implementation work

**We'll find out overnight!** 🌙🚀

---

## Follow-up Work

After experiments complete:

1. **Analyze Results**
   - Create comparison tables and plots
   - Identify best method and configuration

2. **Implement Missing Methods**
   - Build true GRPO training loop
   - Implement iterative Expert Iteration

3. **Scale Up**
   - Train on larger dataset (full MATH dataset)
   - Use larger model (Qwen-7B or Qwen-32B)
   - Evaluate on full test set

4. **Ablation Studies**
   - Test individual components
   - Understand what drives performance

---

**Experiment Status:** RUNNING ✅
**Estimated Completion:** 2025-12-23 18:05 (10 hours from start)
