# Overnight Experiment - Summary & Instructions

## 🌙 What Happened While You Slept

I've set up and launched a comprehensive 10-hour GPU training experiment comparing three alignment methods on the MATH dataset.

**Start Time:** 2025-12-23 08:05:42
**Expected Completion:** 2025-12-23 18:05:42 (~10 hours)
**GPU:** NVIDIA A100 80GB
**Status:** ✅ RUNNING (PID: 61570)

---

## 📊 Experiment Overview

### Three Methods Being Compared (3.33 hours each):

1. **Standard SFT (Supervised Fine-Tuning)**
   - Trains on all 4,999 examples from chat-fmt-base.jsonl
   - Learns from both correct and incorrect model responses

2. **Expert Iteration** (implemented via `--filter-correct`)
   - Trains only on verified correct solutions (~subset of 4,999)
   - Should be more sample-efficient, avoids learning from errors

3. **Advanced SFT** (implemented via `--filter-correct --include-ground-truth`)
   - Trains on correct solutions + ground truth expert solutions
   - Combines model-generated and human-expert patterns

**Note:** True GRPO (Group Relative Policy Optimization) requires a more complex training loop with rollout generation, reward verification, group normalization, and PPO-style updates. This is not yet implemented, so Phase 3 uses an advanced SFT variant as a placeholder.

### Hyperparameter Search Strategy

Each method runs 4 hyperparameter configurations for ~15 minutes each (400 steps), then trains the best configuration for ~140 minutes (2,800 steps).

**Search configurations vary learning rate, warmup ratio, gradient accumulation, and gradient norm clipping.**

---

## 📁 Output Structure

```
overnight_results_20251223_080542/
├── sft/
│   ├── sft_search_01/ to sft_search_04/    # 4 hyperparameter search runs
│   └── final_model/                         # Best config trained for 2,800 steps
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       ├── ...
│       ├── checkpoint-2800/
│       └── metrics.jsonl                    # Training loss over time
│
├── expert_iteration/
│   ├── ei_search_01/ to ei_search_04/
│   └── final_model/
│
├── grpo_sft/
│   ├── grpo_search_01/ to grpo_search_04/
│   └── final_model/
│
├── experiment.log                           # Master log file
└── summary.txt                              # Auto-generated summary
```

---

## 🔍 Monitoring Progress

### Check if experiment is still running:
```bash
ps aux | grep run_overnight_experiments
```

### View main experiment log:
```bash
tail -f overnight_run.log
```

### Check specific training run:
```bash
tail -f overnight_results_20251223_080542/sft/final_model/training.log
```

### View training metrics:
```bash
tail overnight_results_20251223_080542/sft/final_model/metrics.jsonl
# Shows step, loss, learning_rate, etc.
```

### Check GPU usage:
```bash
nvidia-smi
# or watch continuously:
watch -n 5 nvidia-smi
```

### See what checkpoints have been saved:
```bash
ls -lh overnight_results_20251223_080542/*/final_model/checkpoint-*
```

---

## ✅ What To Do When Experiment Completes

### 1. Check Completion Status

The experiment should complete around **18:05** (10 hours from 08:05).

```bash
# Check if process is still running
ps aux | grep run_overnight_experiments

# View final summary
cat overnight_results_20251223_080542/summary.txt

# Check for completion in log
tail -50 overnight_run.log
```

### 2. Analyze Training Results

```bash
# Quick comparison of final losses
for method in sft expert_iteration grpo_sft; do
    echo "=== $method ==="
    tail -1 overnight_results_20251223_080542/$method/final_model/metrics.jsonl
done
```

### 3. Evaluate Models on Benchmark

**Option A: Automatic evaluation script (requires vLLM)**

```bash
# This script will:
# 1. Start vLLM server for each model
# 2. Run eval_math.py on 87 benchmark problems
# 3. Calculate accuracy and generate comparison report

./evaluate_overnight_models.sh overnight_results_20251223_080542
```

The script will create `overnight_results_20251223_080542/evaluations/` with:
- Per-problem results for each method
- Accuracy comparison across methods
- Detailed comparison report

**Option B: Manual evaluation**

```bash
# For each model, you'll need to:
# 1. Start vLLM server
# 2. Run eval_math.py
# 3. Calculate accuracy

# Example for SFT model:
MODEL="overnight_results_20251223_080542/sft/final_model/checkpoint-2800"

# Start vLLM (in separate terminal or screen session)
vllm serve $MODEL \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# Run evaluation (in another terminal)
cd math_rlvr
python eval_math.py \
    --num-samples 87 \
    --jsonl-output sft_final_eval.jsonl \
    --prompt-style chat

# Calculate accuracy
python -c "
import json
with open('sft_final_eval.jsonl') as f:
    results = [json.loads(line) for line in f]
correct = sum(1 for r in results if r['correct'])
total = len(results)
print(f'SFT Accuracy: {correct}/{total} ({correct/total*100:.1f}%)')
"
```

Repeat for `expert_iteration/final_model` and `grpo_sft/final_model`.

---

## 📈 Expected Results

Based on the research literature (DeepSeekMath, DeepSeek-R1):

1. **Expert Iteration ≥ Standard SFT**
   - Training only on correct solutions should improve sample efficiency
   - Model learns good patterns without being contaminated by errors

2. **Advanced SFT (with ground truth) ≥ Expert Iteration**
   - Adding expert-written solutions provides high-quality exemplars
   - Should achieve best accuracy among the three methods

3. **True GRPO would likely outperform all**
   - RL from verified rewards is very powerful
   - But requires additional implementation work

**The experiments will tell us if these hypotheses hold!**

---

## 🎯 Key Questions Answered by This Experiment

1. **Does filtering for correct solutions improve performance?**
   → Compare SFT vs Expert Iteration accuracies

2. **Does adding ground truth solutions help further?**
   → Compare Expert Iteration vs Advanced SFT accuracies

3. **What are the optimal hyperparameters for each method?**
   → Check search_results in each method's directory

4. **How does training data quality affect sample efficiency?**
   → Compare training loss curves (metrics.jsonl files)

5. **Which method produces the strongest model?**
   → Highest benchmark accuracy wins!

---

## 📂 Documentation Created

All experiment details are documented in:

1. **OVERNIGHT_EXPERIMENT_PLAN.md** - Full experiment design, timelines, configurations
2. **OVERNIGHT_RUN_SUMMARY.md** - This file! Quick reference and instructions
3. **overnight_run.log** - Live experiment log (continuously updated)
4. **run_overnight_experiments.sh** - Master experiment script
5. **evaluate_overnight_models.sh** - Post-experiment evaluation script

---

## 🐛 Troubleshooting

### Experiment crashed or stopped early?

Check the log for errors:
```bash
tail -100 overnight_run.log
```

Common issues:
- **CUDA OOM**: Reduce batch size (currently 4)
- **Timeout**: Individual configs have 20-minute timeouts
- **Missing data**: Verify `math_rlvr/chat-fmt-base.jsonl` exists

### Want to restart or continue?

The script supports timeouts, so partial results should be saved. You can manually run the final training phase for any method:

```bash
source .venv/bin/activate

python sft_qwen_math.py \
    --train-data math_rlvr/chat-fmt-base.jsonl \
    --model-name Qwen/Qwen2.5-Math-1.5B \
    --output-dir overnight_results_20251223_080542/sft/final_model \
    --max-steps 2800 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.2 \
    --gradient-accumulation-steps 2 \
    --max-grad-norm 1.0 \
    --batch-size 4 \
    --save-steps 500 \
    --logging-steps 20 \
    --bf16 \
    --resume-from overnight_results_20251223_080542/sft/final_model/checkpoint-XXXX
```

---

## 🚀 Next Steps After Experiment

1. **Evaluate models** using the evaluation script
2. **Compare results** across all three methods
3. **Analyze what worked**:
   - Best hyperparameters for each method
   - Training curves and convergence behavior
   - Accuracy vs training loss correlation

4. **Implement True GRPO** (optional):
   - Build rollout generation pipeline
   - Implement group-wise reward normalization
   - Add PPO-style policy gradient updates
   - Compare against SFT baselines

5. **Scale up** (optional):
   - Train on full MATH dataset (~12K problems)
   - Use larger model (Qwen-7B or Qwen-32B)
   - Evaluate on full MATH test set (5K problems)

---

## 💡 Notes

### What's Already Implemented ✅
- Supervised Fine-Tuning (SFT) infrastructure
- GRPO unit tests (all 14 tests passing!)
- Math verification system (math_verify)
- Evaluation framework (eval_math.py)
- Hyperparameter sweep utilities

### What's Not Yet Implemented ⏳
- True GRPO training loop (rollouts + RL updates)
- Iterative Expert Iteration (generate → verify → retrain loop)
- DPO (Direct Preference Optimization)
- Full safety/RLHF suite

### Experiment Design Decisions

**Why these hyperparameter ranges?**
- Based on previous sweep results in `hparam_sweep_analysis.md`
- LR 5e-5 with warmup 0.2 performed best in initial SFT tests
- Conservative LRs (1e-5 to 5e-5) for math reasoning tasks
- Gradient accumulation (2-4) to simulate larger batch sizes

**Why 400 steps for search, 2800 for final?**
- 400 steps ≈ 15 minutes, enough to see initial convergence
- 2800 steps ≈ 140 minutes, substantial training on 4999 examples
- Multiple epochs over dataset with checkpointing

**Why bf16 (bfloat16) precision?**
- A100 GPU has excellent bf16 support
- Maintains numerical stability for LLM training
- 2x memory savings vs fp32, enabling larger batch sizes

---

## 📊 Preliminary Results (Available During Training)

You can check preliminary results even while training is ongoing:

```bash
# SFT Phase (should be done after ~3.3 hours)
echo "SFT hyperparameter search results:"
for i in {1..4}; do
    run="overnight_results_20251223_080542/sft/sft_search_0$i"
    if [ -f "$run/metrics.jsonl" ]; then
        loss=$(tail -1 "$run/metrics.jsonl" | python -c "import json,sys; print(json.load(sys.stdin).get('loss', 'N/A'))")
        echo "  Config $i: final loss = $loss"
    fi
done

# Expert Iteration Phase (should be done after ~6.6 hours)
echo "\nExpert Iteration hyperparameter search results:"
for i in {1..4}; do
    run="overnight_results_20251223_080542/expert_iteration/ei_search_0$i"
    if [ -f "$run/metrics.jsonl" ]; then
        loss=$(tail -1 "$run/metrics.jsonl" | python -c "import json,sys; print(json.load(sys.stdin).get('loss', 'N/A'))")
        echo "  Config $i: final loss = $loss"
    fi
done
```

---

## 🎓 What You're Learning

This experiment demonstrates:

1. **Hyperparameter optimization** - Systematic search for best training configurations
2. **Alignment methods** - Different approaches to teaching LLMs (SFT, Expert Iteration, RL)
3. **Sample efficiency** - How data quality affects learning speed
4. **Production ML practices** - Long-running experiments, checkpointing, monitoring, evaluation

---

**Happy model training! 🚀**

Check back in ~10 hours to see which method produces the strongest math problem solver!

---

**Quick Status Check:**
```bash
# Run this command to get current status:
echo "Experiment started: 2025-12-23 08:05:42"
echo "Current time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Elapsed: $(ps -o etime= -p 61570 2>/dev/null || echo 'Process not found')"
echo "GPU usage:"
nvidia-smi | grep -A 2 "GPU  Name"
```
