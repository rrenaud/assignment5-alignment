# Overnight GPU Experiments: Comprehensive Guide
## Assignment 5 - Alignment Methods for Mathematical Reasoning

**Version:** 1.0
**Created:** 2025-12-24
**Status:** Experiments running (started 2025-12-23 08:11:39)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Scientific Motivation](#scientific-motivation)
3. [Experimental Design](#experimental-design)
4. [Inputs Specification](#inputs-specification)
5. [Outputs Specification](#outputs-specification)
6. [Interpretation Guide](#interpretation-guide)
7. [Post-Experiment Analysis](#post-experiment-analysis)
8. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### What Are We Testing?
This overnight experiment compares **three alignment methods** for improving a language model's mathematical reasoning ability:

1. **Standard SFT** - Learn from all examples (baseline)
2. **Expert Iteration** - Learn only from correct solutions
3. **Advanced SFT** - Learn from correct solutions + expert ground truth

### Why Does This Matter?
Recent research (DeepSeekMath, DeepSeek-R1) shows that training on **verified correct solutions** dramatically improves mathematical reasoning. We're empirically testing whether this holds for our setup.

### Key Hypothesis
**Expert Iteration ≥ Standard SFT** and **Advanced SFT ≥ Expert Iteration**

The hypothesis is that filtering out incorrect solutions prevents the model from learning flawed reasoning patterns, leading to better generalization.

### Timeline
- **Total Duration:** 10 hours (600 minutes)
- **Per Method:** 3.33 hours
  - Hyperparameter search: 1 hour (4 configs)
  - Final training: 2.33 hours (~2,800 steps)

---

## Scientific Motivation

### Background: Why Expert Iteration?

**Traditional SFT Problem:**
When training on model-generated solutions, the model may learn from both correct and incorrect reasoning patterns. This can:
- Reinforce errors and hallucinations
- Create inconsistent reasoning behavior
- Reduce sample efficiency

**Expert Iteration Solution:**
By filtering to only train on **verified correct solutions**, we:
- Eliminate contamination from incorrect reasoning
- Increase effective data quality
- Potentially achieve better performance with fewer examples

**Theoretical Foundation:**
This approach is inspired by:
- **DeepSeekMath** (arXiv:2402.03300): Achieved 51.7% on MATH benchmark using RL with verified rewards
- **DeepSeek-R1** (arXiv:2501.12948): Extended this to general reasoning tasks
- **AlphaGo/AlphaZero**: Original expert iteration for game playing

### Three Methods Explained

#### 1. Standard SFT (Baseline)
**What:** Train on all 4,999 examples regardless of correctness

**Training Data:**
- All model-generated solutions from Qwen3-32B
- Mix of correct and incorrect solutions
- Total: 4,999 problem-solution pairs

**Expected Behavior:**
- Model learns to imitate the response format
- May learn both correct and incorrect reasoning patterns
- Serves as baseline for comparison

**Command:**
```bash
python sft_qwen_math.py \
    --train-data math_rlvr/chat-fmt-base.jsonl \
    --output-dir results/sft/final_model \
    --max-steps 2800
```

#### 2. Expert Iteration
**What:** Train only on verified correct solutions

**Training Data:**
- Subset of examples where model-generated solution is correct
- Filtered using `math_verify` grader
- Estimated: ~1,600-2,000 correct examples (33-40% of total)

**Expected Behavior:**
- Model learns only from successful problem-solving patterns
- Higher data quality but smaller dataset
- Should achieve better accuracy despite fewer examples

**Command:**
```bash
python sft_qwen_math.py \
    --train-data math_rlvr/chat-fmt-base.jsonl \
    --filter-correct \
    --output-dir results/expert_iteration/final_model \
    --max-steps 2800
```

**Key Difference:** The `--filter-correct` flag filters the dataset during loading to include only examples where `correct=true`.

#### 3. Advanced SFT
**What:** Train on correct solutions + ground truth expert solutions

**Training Data:**
- All verified correct model solutions
- PLUS original MATH dataset ground truth solutions
- Augmented dataset with high-quality expert reasoning

**Expected Behavior:**
- Benefits from both model-generated and expert solutions
- Ground truth provides gold-standard reasoning patterns
- Should achieve best performance (hypothesis)

**Command:**
```bash
python sft_qwen_math.py \
    --train-data math_rlvr/chat-fmt-base.jsonl \
    --filter-correct \
    --include-ground-truth \
    --output-dir results/grpo_sft/final_model \
    --max-steps 2800
```

**Key Differences:**
- `--filter-correct`: Only correct model solutions
- `--include-ground-truth`: Adds expert solutions from MATH dataset

---

## Experimental Design

### Phase Structure

Each of the three methods follows the same two-phase structure:

#### Phase 1: Hyperparameter Search (1 hour)
**Goal:** Find the best learning configuration for each method

**Process:**
1. Test 4 hyperparameter configurations
2. Train each for 400 steps (~15 minutes)
3. Select configuration with lowest final loss
4. Use best config for Phase 2

**Why?** Different alignment methods may require different hyperparameters:
- Expert Iteration has less data → may need higher LR
- Advanced SFT has higher quality data → may need lower LR

#### Phase 2: Final Training (2.33 hours)
**Goal:** Train best model to convergence

**Process:**
1. Use best hyperparameters from Phase 1
2. Train for 2,800 steps
3. Save checkpoints every 500 steps
4. Monitor convergence via loss metrics

### Hyperparameter Search Space

#### Standard SFT & Expert Iteration
```python
configs = [
    {"lr": 1e-5,  "warmup": 0.1,  "grad_accum": 2, "grad_norm": 0.5},
    {"lr": 2e-5,  "warmup": 0.1,  "grad_accum": 2, "grad_norm": 1.0},
    {"lr": 5e-5,  "warmup": 0.2,  "grad_accum": 2, "grad_norm": 1.0},
    {"lr": 3e-5,  "warmup": 0.15, "grad_accum": 4, "grad_norm": 1.0},
]
```

**Rationale:**
- Learning rate range: 1e-5 to 5e-5 (standard for fine-tuning)
- Warmup: 10-20% of training (prevents early instability)
- Gradient accumulation: 2-4 steps (effective batch size 2-4)
- Gradient clipping: 0.5-1.0 (prevents exploding gradients)

#### Advanced SFT (More Conservative)
```python
configs = [
    {"lr": 5e-6,  "warmup": 0.1,  "grad_accum": 4, "grad_norm": 0.5},
    {"lr": 1e-5,  "warmup": 0.1,  "grad_accum": 4, "grad_norm": 0.5},
    {"lr": 2e-5,  "warmup": 0.15, "grad_accum": 4, "grad_norm": 1.0},
    {"lr": 3e-5,  "warmup": 0.2,  "grad_accum": 4, "grad_norm": 1.0},
]
```

**Rationale:**
- Lower learning rates (5e-6 to 3e-5)
- Higher quality data may need gentler updates
- Increased gradient accumulation for stability

### Fixed Training Parameters

**Across All Experiments:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Qwen/Qwen2.5-Math-1.5B | Pre-trained math specialist, good balance of size/performance |
| Batch Size | 1 | Memory constraint (54GB already used) |
| Precision | bfloat16 | Memory efficiency + numerical stability |
| Optimizer | AdamW | Standard for transformer fine-tuning |
| Weight Decay | 0.01 | Regularization |
| Max Sequence Length | 2048 | Accommodates long reasoning chains |

**Why Batch Size = 1?**
- GPU had 54GB/81GB already occupied by unknown processes
- Reduced from initial plan of batch_size=4
- Compensated with gradient accumulation (effective batch size = batch_size × grad_accum)

---

## Inputs Specification

### Training Data

#### Primary Dataset: `math_rlvr/chat-fmt-base.jsonl`

**Source:** MATH dataset problems with Qwen3-32B generated solutions

**Format:**
```json
{
  "problem": "Simplify $\\frac{x+1}{3}+\\frac{2-3x}{2}$. Express your answer as...",
  "ground_truth": "Using a common denominator, $\\frac{x+1}{3}+\\frac{2-3x}{2}=...",
  "predicted": "<think>To simplify this expression...</think><answer>\\boxed{-7x+8}</answer>",
  "correct": true,
  "level": "Level 3",
  "subject": "Algebra",
  "response": "<think>Let me work through this step by step...</think>..."
}
```

**Fields:**
- `problem`: Original MATH dataset problem statement
- `ground_truth`: Expert solution from MATH dataset
- `predicted`: Final answer extracted from model response
- `correct`: Boolean indicating if predicted answer matches ground truth
- `level`: Difficulty (Level 1-5)
- `subject`: Math topic (Algebra, Geometry, Number Theory, etc.)
- `response`: Full model-generated solution with reasoning

**Statistics:**
- Total examples: 4,999
- Estimated correct: ~1,673 (33.5%)
- Average response length: ~300-500 tokens
- Contains `<think>` tags for reasoning, `<answer>` tags with `\boxed{}` for final answer

#### Ground Truth Augmentation

When `--include-ground-truth` is used:
- Each problem gets TWO training examples:
  1. Model-generated correct solution (if correct=true)
  2. Original MATH dataset expert solution

**Format for ground truth:**
```json
{
  "problem": "Simplify $\\frac{x+1}{3}+\\frac{2-3x}{2}$...",
  "response": "Using a common denominator, $\\frac{x+1}{3}+\\frac{2-3x}{2}=..."
}
```

### Evaluation Data

#### Benchmark: `math_rlvr/benchmark_dataset.jsonl`

**Purpose:** Held-out test set for final model evaluation

**Statistics:**
- Total examples: 87
- Stratified by difficulty and subject
- Same format as training data

**Not Used During Training:** This is strictly for post-experiment evaluation

### Model Inputs

#### Base Model: Qwen/Qwen2.5-Math-1.5B

**Specifications:**
- Parameters: 1.54 billion
- Architecture: Decoder-only transformer
- Vocabulary: 151,643 tokens
- Context Length: 131,072 tokens (we use max 2,048)
- Pre-training: Mathematics-focused corpus

**Why This Model?**
- Specialized for mathematical reasoning
- Good balance of size and performance
- Fits in available GPU memory with reasonable batch size

**Tokenizer:**
- Type: BPE (Byte-Pair Encoding)
- Special tokens: `<|im_start|>`, `<|im_end|>` for chat format
- Padding token: 151643

---

## Outputs Specification

### Directory Structure

```
overnight_results_20251223_081139/
│
├── experiment.log              # Master log file (all phases)
├── summary.txt                 # Auto-generated final summary
│
├── sft/                        # Phase 1: Standard SFT
│   ├── sft_search_01/          # Config 1: LR=1e-5
│   │   ├── metrics_20251223_081206.jsonl
│   │   ├── training.log
│   │   ├── checkpoint-epoch0-step200/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── generation_config.json
│   │   │   └── training_state.pt
│   │   ├── checkpoint-epoch0-step400/
│   │   └── tokenizer files
│   ├── sft_search_02/          # Config 2: LR=2e-5
│   ├── sft_search_03/          # Config 3: LR=5e-5
│   ├── sft_search_04/          # Config 4: LR=3e-5
│   └── final_model/            # Best config, 2800 steps
│       ├── checkpoint-epoch0-step500/
│       ├── checkpoint-epoch0-step1000/
│       ├── checkpoint-epoch0-step1500/
│       ├── checkpoint-epoch0-step2000/
│       ├── checkpoint-epoch0-step2500/
│       ├── checkpoint-epoch0-step2800/
│       └── metrics_*.jsonl
│
├── expert_iteration/           # Phase 2: Expert Iteration
│   ├── ei_search_01/ to ei_search_04/
│   └── final_model/
│
└── grpo_sft/                   # Phase 3: Advanced SFT
    ├── grpo_search_01/ to grpo_search_04/
    └── final_model/
```

### File Formats

#### 1. `metrics.jsonl` - Training Metrics

**Purpose:** Per-step training metrics for analysis

**Format:** One JSON object per line
```json
{
  "loss/train": 0.0943,
  "learning_rate": 9.999951e-06,
  "gradient_norm": 1.4765625,
  "epoch": 1,
  "loss/total": 0.1099,
  "loss/perplexity": 1.1162,
  "accuracy/token": 0.9919,
  "loss/per_token_avg": 0.1100,
  "loss/boxed_tokens": 0.0000148,
  "stats/boxed_token_count": 6.0,
  "accuracy/boxed_tokens": 1.0,
  "loss/thinking_tokens": 0.0,
  "stats/thinking_token_count": 0.0,
  "loss/other_tokens": 0.1115,
  "stats/other_token_count": 437.5,
  "stats/total_tokens": 443.5,
  "stats/batch_size": 1.0,
  "stats/boxed_pct": 1.34,
  "stats/thinking_pct": 0.0,
  "stats/other_pct": 98.66,
  "step": 10,
  "timestamp": "2025-12-23T08:12:19.001527"
}
```

**Key Metrics:**
- `loss/train`: Primary training loss (negative log-likelihood)
- `loss/boxed_tokens`: Loss on final answer tokens (most important)
- `loss/thinking_tokens`: Loss on reasoning chain tokens
- `loss/other_tokens`: Loss on other response tokens
- `accuracy/token`: Token-level prediction accuracy
- `gradient_norm`: Gradient magnitude (for monitoring stability)

**Logged Every:**
- 10 steps for basic metrics
- 10 steps for detailed component breakdown

#### 2. `training.log` - Human-Readable Output

**Purpose:** Real-time training progress with tqdm-style progress bar

**Example:**
```
Training:   5%|▍         | 262/5019 [00:10:45<3:15:22, 2.39it/s, loss=0.0495, lr=4.89e-05, grad=0.97, epoch=1, boxed=0.000]
```

**Contains:**
- Progress percentage
- Current step / total steps
- Elapsed time and ETA
- Training speed (iterations/second)
- Current loss, learning rate, gradient norm
- Epoch number
- Boxed token metrics

#### 3. `checkpoint-epochX-stepY/` - Model Checkpoints

**Purpose:** Saved model states for resuming or evaluation

**Contents:**
- `config.json`: Model architecture configuration
- `model.safetensors`: Model weights (safe format)
- `generation_config.json`: Generation hyperparameters
- `training_state.pt`: Optimizer + scheduler state (for resuming)
- Tokenizer files (copied from base model)

**Size:** ~3GB per checkpoint for 1.5B model

**Saved At:**
- Search phase: Every 200 steps
- Final training: Every 500 steps
- Final step always saved

#### 4. `experiment.log` - Master Log

**Purpose:** High-level experiment orchestration log

**Contains:**
```
[2025-12-23 08:11:39] ========================================================================
[2025-12-23 08:11:39] OVERNIGHT GPU EXPERIMENTS - ASSIGNMENT 5
[2025-12-23 08:11:39] ========================================================================
[2025-12-23 08:11:39] Step 1: Hyperparameter search (4 configs x 15 mins = 60 mins)
[2025-12-23 08:11:39]   Config 1/4: LR=1e-5, warmup=0.1, grad_accum=2, grad_norm=0.5
[2025-12-23 08:26:45]   Config 1 completed. Final loss: 0.0876
[2025-12-23 08:26:45]   Config 2/4: LR=2e-5, warmup=0.1, grad_accum=2, grad_norm=1.0
...
[2025-12-23 11:31:22] Best SFT config: 5e-5:0.2:2:1.0 (loss: 0.0543)
[2025-12-23 11:31:22] Step 2: Training final SFT model with best config (~140 mins)
...
```

**Use Case:** Quick overview of experiment progression

#### 5. `summary.txt` - Final Summary (Auto-Generated)

**Purpose:** Experiment recap created at completion

**Contents:**
```
Overnight Experiment Summary
============================

Start time: 2025-12-23 08:11:39
End time: 2025-12-23 18:11:39
Total duration: 10 hours

Models trained:
1. SFT: overnight_results_20251223_081139/sft/final_model
2. Expert Iteration: overnight_results_20251223_081139/expert_iteration/final_model
3. Advanced SFT: overnight_results_20251223_081139/grpo_sft/final_model

Best hyperparameters:
- SFT: LR=5e-5, warmup=0.2, grad_accum=2, grad_norm=1.0
- Expert Iteration: LR=3e-5, warmup=0.15, grad_accum=2, grad_norm=1.0
- Advanced SFT: LR=2e-5, warmup=0.15, grad_accum=4, grad_norm=1.0

Final training losses:
- SFT: 0.543
- Expert Iteration: 0.487
- Advanced SFT: 0.421

Next steps: Run evaluation on benchmark dataset
```

---

## Interpretation Guide

### What Do the Metrics Mean?

#### Training Loss

**Definition:** Negative log-likelihood of correct tokens given context

**Interpretation:**
- **Lower is better**
- Initial loss (untrained): ~2.0-3.0
- Good final loss: ~0.5-0.8
- Excellent final loss: ~0.3-0.5

**Formula:**
```
loss = -mean(log P(token_i | context))
```

Where mean is over all response tokens (prompt tokens excluded).

**What It Measures:**
- How confidently the model predicts the correct next token
- Lower loss = model is more certain about correct predictions

**Typical Progression:**
```
Step    Loss    Interpretation
0       2.3     Random initialization
100     1.5     Learning basic patterns
500     0.9     Understanding problem structure
1000    0.7     Good reasoning ability
2800    0.5     Near-convergence
```

#### Component Losses

The loss is decomposed into three components:

**1. Boxed Token Loss** (`loss/boxed_tokens`)
- **What:** Loss on final answer tokens inside `\boxed{}`
- **Why Important:** This is what we ultimately care about - getting the final answer right
- **Target:** As close to 0 as possible
- **Typical Range:** 0.0-0.1 (much lower than other components)

**2. Thinking Token Loss** (`loss/thinking_tokens`)
- **What:** Loss on reasoning chain inside `<think>...</think>`
- **Why Important:** Good reasoning leads to correct answers
- **Target:** ~0.3-0.6
- **Typical Range:** 0.2-0.8

**3. Other Token Loss** (`loss/other_tokens`)
- **What:** Loss on explanation tokens, formatting, etc.
- **Why Important:** Less critical, but affects response quality
- **Target:** ~0.5-0.7
- **Typical Range:** 0.4-1.0

**Ideal Pattern:**
```
boxed_tokens << thinking_tokens < other_tokens
```

This means the model is most certain about final answers, moderately certain about reasoning, and less certain about formatting.

#### Gradient Norm

**Definition:** Magnitude of gradient vector during backpropagation

**Interpretation:**
- **Too high (>5.0):** Risk of exploding gradients, instability
- **Optimal (0.5-2.0):** Healthy learning
- **Too low (<0.1):** Vanishing gradients, slow learning

**What We Do:** Clip gradients at `max_grad_norm` (0.5-1.0) to prevent instability

**Monitoring:**
```
If gradient_norm consistently hits max_grad_norm:
  → Gradients are being clipped frequently
  → May need lower learning rate or different hyperparameters
```

#### Learning Rate Schedule

**Type:** Cosine decay with warmup

**Phases:**
1. **Warmup** (0-10% of training): Linear increase from 0 to max_lr
2. **Decay** (10-100%): Cosine decay from max_lr to 0

**Why?**
- Warmup prevents early instability
- Cosine decay allows fine-grained optimization near convergence

**Formula:**
```python
if step < warmup_steps:
    lr = max_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = 0.5 * max_lr * (1 + cos(π * progress))
```

### How to Compare Methods

#### 1. Training Loss Comparison

**What to Look For:**
- Which method achieves lowest final loss?
- Which converges fastest (fewer steps to good loss)?
- How stable is training (smooth loss curve vs. spiky)?

**Expected Pattern (Hypothesis):**
```
Standard SFT: final_loss ~ 0.6-0.7
Expert Iteration: final_loss ~ 0.5-0.6  (10-15% lower)
Advanced SFT: final_loss ~ 0.4-0.5      (20-30% lower)
```

**Why?**
- Expert Iteration: Higher quality data → lower loss
- Advanced SFT: Expert solutions + model solutions → even lower loss

**How to Check:**
```bash
# Get final loss for each method
for method in sft expert_iteration grpo_sft; do
    echo "=== $method ==="
    tail -1 overnight_results_20251223_081139/$method/final_model/metrics_*.jsonl | \
        python -c "import json,sys; print(f\"Final loss: {json.load(sys.stdin)['loss/train']:.4f}\")"
done
```

#### 2. Benchmark Accuracy Comparison

**What to Look For:**
- % of problems solved correctly on held-out test set
- Performance by difficulty level
- Performance by subject area

**Expected Pattern (Hypothesis):**
```
Standard SFT:       ~30-35% accuracy
Expert Iteration:   ~40-45% accuracy  (+30% relative improvement)
Advanced SFT:       ~50-55% accuracy  (+20% over EI, +60% over SFT)
```

**Why This Matters More Than Loss:**
- Loss is a proxy metric
- Accuracy is what we actually care about
- A model with slightly higher loss might still have better accuracy

**How to Evaluate:**
```bash
# After training completes, run evaluation (requires vLLM)
cd math_rlvr

# Evaluate SFT model
python eval_math.py \
    --model-path ../overnight_results_20251223_081139/sft/final_model \
    --split test \
    --num-samples 87 \
    --jsonl-output sft_eval.jsonl

# Repeat for other methods
# Then compare accuracies
```

#### 3. Data Efficiency

**What to Look For:**
- How many training examples did each method use?
- Loss per example (sample efficiency)

**Expected Pattern:**
```
Standard SFT:       4,999 examples, loss ~0.65
Expert Iteration:   ~1,673 examples, loss ~0.55
Advanced SFT:       ~3,346 examples, loss ~0.45

Sample efficiency (loss / num_examples):
Standard SFT:       0.00013
Expert Iteration:   0.00033  (2.5x more efficient)
Advanced SFT:       0.00013  (similar to SFT, but better absolute loss)
```

**Insight:** Expert Iteration is more sample-efficient (learns more from less data)

### Hyperparameter Analysis

#### What Makes a Good Configuration?

**Criteria:**
1. **Lowest final loss** (primary metric for search)
2. **Stable training** (smooth loss curve, no spikes)
3. **Fast convergence** (reaches low loss in fewer steps)

#### How to Interpret Search Results

**Example Search Output:**
```
Config 1 (LR=1e-5):  final_loss=0.0876, stable=yes, slow=yes
Config 2 (LR=2e-5):  final_loss=0.0654, stable=yes, medium=yes
Config 3 (LR=5e-5):  final_loss=0.0543, stable=yes, fast=yes  ← BEST
Config 4 (LR=3e-5):  final_loss=0.0612, stable=no, fast=yes
```

**Decision:** Choose Config 3
- **Why?** Lowest final loss + stable + fast convergence
- **Trade-off:** Config 4 trains faster but is unstable (worse than Config 3's lower loss)

#### Learning Rate Sensitivity

**Pattern to Expect:**
```
LR too low (1e-6):   Slow convergence, may not reach optimal loss
LR optimal (1e-5 to 5e-5):  Good balance of speed and stability
LR too high (1e-4):  Unstable, loss spikes, may diverge
```

**How to Identify Optimal LR:**
1. Plot loss vs. step for each config
2. Look for "knee" in the curve (fastest descent without instability)
3. Check gradient norms (should be 0.5-2.0, not hitting max clip frequently)

---

## Post-Experiment Analysis

### Step 1: Verify Completion

```bash
# Check if all models completed
echo "Checking experiment completion..."

# Expected: 3 final model directories with checkpoint-2800
for method in sft expert_iteration grpo_sft; do
    final_ckpt="overnight_results_20251223_081139/$method/final_model/checkpoint-epoch0-step2800"
    if [ -d "$final_ckpt" ]; then
        echo "✓ $method: COMPLETE"
    else
        echo "✗ $method: INCOMPLETE"
    fi
done

# Check total training time
start_time=$(head -1 overnight_results_20251223_081139/experiment.log | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
end_time=$(tail -1 overnight_results_20251223_081139/experiment.log | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
echo "Start: $start_time"
echo "End: $end_time"
```

### Step 2: Extract Training Metrics

```bash
# Create analysis directory
mkdir -p analysis

# Extract final losses for each method
python3 << 'EOF'
import json
from pathlib import Path

results_dir = Path("overnight_results_20251223_081139")
methods = ["sft", "expert_iteration", "grpo_sft"]

print("=" * 60)
print("FINAL TRAINING LOSSES")
print("=" * 60)

for method in methods:
    metrics_file = list((results_dir / method / "final_model").glob("metrics_*.jsonl"))[0]

    with open(metrics_file) as f:
        lines = f.readlines()
        final_metrics = json.loads(lines[-1])

    print(f"\n{method.upper()}:")
    print(f"  Final loss: {final_metrics['loss/train']:.4f}")
    print(f"  Boxed loss: {final_metrics.get('loss/boxed_tokens', 'N/A')}")
    print(f"  Steps: {final_metrics['step']}")
    print(f"  Learning rate: {final_metrics['learning_rate']:.2e}")
EOF
```

### Step 3: Plot Training Curves

```bash
# Requires matplotlib
python3 << 'EOF'
import json
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("overnight_results_20251223_081139")
methods = ["sft", "expert_iteration", "grpo_sft"]
colors = ["blue", "green", "orange"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for idx, method in enumerate(methods):
    metrics_file = list((results_dir / method / "final_model").glob("metrics_*.jsonl"))[0]

    steps, losses, boxed_losses, lrs = [], [], [], []

    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            steps.append(m['step'])
            losses.append(m['loss/train'])
            if 'loss/boxed_tokens' in m:
                boxed_losses.append(m['loss/boxed_tokens'])
            lrs.append(m['learning_rate'])

    # Plot overall loss
    axes[0, 0].plot(steps, losses, label=method, color=colors[idx])

    # Plot boxed loss
    if boxed_losses:
        axes[0, 1].plot(steps[:len(boxed_losses)], boxed_losses,
                       label=method, color=colors[idx])

    # Plot learning rate
    axes[1, 0].plot(steps, lrs, label=method, color=colors[idx])

    # Plot last 500 steps (zoomed)
    axes[1, 1].plot(steps[-500:], losses[-500:], label=method, color=colors[idx])

axes[0, 0].set_title("Overall Training Loss")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].set_title("Boxed Token Loss (Final Answer)")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].set_title("Learning Rate Schedule")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Learning Rate")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].set_title("Training Loss (Last 500 Steps)")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("analysis/training_curves.png", dpi=150)
print("Saved plot to analysis/training_curves.png")
EOF
```

### Step 4: Evaluate on Benchmark

**Prerequisites:**
- vLLM installed
- Sufficient GPU memory (~40GB for vLLM + model)

```bash
cd math_rlvr

# Function to evaluate a model
evaluate_model() {
    local method=$1
    local model_path="../overnight_results_20251223_081139/$method/final_model"

    echo "Evaluating $method..."

    # Start vLLM server (requires separate terminal or background process)
    # vllm serve $model_path --port 8000 &
    # VLLM_PID=$!
    # sleep 30  # Wait for server to start

    # Run evaluation
    python eval_math.py \
        --model-path $model_path \
        --split test \
        --num-samples 87 \
        --jsonl-output "${method}_eval.jsonl" \
        --prompt-style chat

    # Kill vLLM server
    # kill $VLLM_PID
}

# Evaluate all three methods
evaluate_model "sft"
evaluate_model "expert_iteration"
evaluate_model "grpo_sft"
```

### Step 5: Compare Results

```bash
python3 << 'EOF'
import json
from pathlib import Path
from collections import defaultdict

methods = ["sft", "expert_iteration", "grpo_sft"]
results = {}

print("=" * 80)
print("BENCHMARK EVALUATION RESULTS")
print("=" * 80)

for method in methods:
    eval_file = f"math_rlvr/{method}_eval.jsonl"

    with open(eval_file) as f:
        evals = [json.loads(line) for line in f]

    total = len(evals)
    correct = sum(1 for e in evals if e['correct'])
    accuracy = correct / total * 100

    # Breakdown by level
    by_level = defaultdict(lambda: {"correct": 0, "total": 0})
    for e in evals:
        level = e.get('level', 'Unknown')
        by_level[level]['total'] += 1
        if e['correct']:
            by_level[level]['correct'] += 1

    results[method] = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'by_level': dict(by_level)
    }

    print(f"\n{method.upper()}")
    print(f"  Overall: {correct}/{total} = {accuracy:.1f}%")
    print(f"  By Level:")
    for level in sorted(by_level.keys()):
        level_acc = by_level[level]['correct'] / by_level[level]['total'] * 100
        print(f"    {level}: {by_level[level]['correct']}/{by_level[level]['total']} = {level_acc:.1f}%")

# Relative improvements
print("\n" + "=" * 80)
print("RELATIVE IMPROVEMENTS")
print("=" * 80)

sft_acc = results['sft']['accuracy']
ei_acc = results['expert_iteration']['accuracy']
adv_acc = results['grpo_sft']['accuracy']

print(f"\nExpert Iteration vs. SFT:")
print(f"  Absolute: +{ei_acc - sft_acc:.1f} percentage points")
print(f"  Relative: +{(ei_acc / sft_acc - 1) * 100:.1f}%")

print(f"\nAdvanced SFT vs. Expert Iteration:")
print(f"  Absolute: +{adv_acc - ei_acc:.1f} percentage points")
print(f"  Relative: +{(adv_acc / ei_acc - 1) * 100:.1f}%")

print(f"\nAdvanced SFT vs. SFT:")
print(f"  Absolute: +{adv_acc - sft_acc:.1f} percentage points")
print(f"  Relative: +{(adv_acc / sft_acc - 1) * 100:.1f}%")
EOF
```

### Step 6: Statistical Significance

```bash
# Test if differences are statistically significant
python3 << 'EOF'
import json
from scipy.stats import chi2_contingency
import numpy as np

methods = ["sft", "expert_iteration", "grpo_sft"]
results = []

# Load results
for method in methods:
    with open(f"math_rlvr/{method}_eval.jsonl") as f:
        results.append([json.loads(line)['correct'] for line in f])

# Create contingency table
sft_results = results[0]
ei_results = results[1]
adv_results = results[2]

# SFT vs. Expert Iteration
contingency_sft_ei = [
    [sum(sft_results), len(sft_results) - sum(sft_results)],
    [sum(ei_results), len(ei_results) - sum(ei_results)]
]
chi2, p_value_sft_ei, _, _ = chi2_contingency(contingency_sft_ei)

# Expert Iteration vs. Advanced SFT
contingency_ei_adv = [
    [sum(ei_results), len(ei_results) - sum(ei_results)],
    [sum(adv_results), len(adv_results) - sum(adv_results)]
]
chi2, p_value_ei_adv, _, _ = chi2_contingency(contingency_ei_adv)

print("Statistical Significance (Chi-square test):")
print(f"  SFT vs. Expert Iteration: p = {p_value_sft_ei:.4f}")
print(f"  Expert Iteration vs. Advanced SFT: p = {p_value_ei_adv:.4f}")
print(f"\nInterpretation:")
print(f"  p < 0.05: Statistically significant")
print(f"  p < 0.01: Highly significant")
print(f"  p >= 0.05: Not statistically significant")
EOF
```

---

## Troubleshooting

### Common Issues During Training

#### 1. CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Causes:**
- Batch size too large
- Sequence length too long
- Gradient accumulation not working

**Solutions:**
```bash
# Reduce batch size
--batch-size 1  # Already at minimum

# Reduce max sequence length
--max-length 1536  # Down from 2048

# Increase gradient accumulation
--gradient-accumulation-steps 8  # Up from 2-4
```

#### 2. Loss Not Decreasing

**Symptoms:**
- Loss stays at initial value (~2.0)
- Loss oscillates wildly
- Loss increases over time

**Causes:**
- Learning rate too high/low
- Data quality issues
- Model/data mismatch

**Diagnostic Steps:**
```bash
# Check learning rate
tail -100 training.log | grep "lr="

# Check gradient norms
tail -100 training.log | grep "grad="

# Verify data is loading correctly
python3 << 'EOF'
import json
with open("math_rlvr/chat-fmt-base.jsonl") as f:
    sample = json.loads(f.readline())
    print(json.dumps(sample, indent=2))
EOF
```

**Solutions:**
- Try lower learning rate (1e-6)
- Check data format matches expectations
- Verify model loaded correctly

#### 3. Training Speed Too Slow

**Symptoms:**
- < 1 iteration/second
- Much slower than expected

**Causes:**
- CPU bottleneck in data loading
- Excessive logging
- GPU utilization low

**Solutions:**
```bash
# Increase dataloader workers
--num-workers 8  # Up from 4

# Reduce logging frequency
--logging-steps 50  # Up from 10

# Check GPU utilization
nvidia-smi dmon -s u -d 1
```

#### 4. Experiment Stopped Prematurely

**Symptoms:**
- Process not running
- Incomplete checkpoints

**Diagnostic:**
```bash
# Check if process is running
ps aux | grep run_overnight_experiments

# Check last lines of log
tail -100 overnight_run.log

# Look for errors
grep -i error overnight_run.log
grep -i fail overnight_run.log
```

**Common Causes:**
- Timeout reached (20 min per config)
- CUDA error
- Disk full
- Manual interruption

### Monitoring During Training

#### Real-Time Progress

```bash
# Follow master log
tail -f overnight_run.log

# Follow specific training run
tail -f overnight_results_20251223_081139/sft/sft_search_01/training.log

# Watch GPU usage
watch -n 2 nvidia-smi
```

#### Check Current Status

```bash
# Get experiment status summary
echo "=== EXPERIMENT STATUS ==="

# Check if running
if ps aux | grep -q "[r]un_overnight_experiments"; then
    echo "Status: RUNNING"

    # Get runtime
    ps -o etime= -p $(pgrep -f run_overnight_experiments)
else
    echo "Status: STOPPED"
fi

# GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

# Current phase
tail -1 overnight_run.log | grep -oP '\[.*?\] .*'

# Latest checkpoint
ls -lt overnight_results_20251223_081139/*/final_model/checkpoint-* 2>/dev/null | head -1
```

#### Estimate Time Remaining

```bash
# Calculate remaining time
python3 << 'EOF'
from datetime import datetime, timedelta
import subprocess

# Get start time from log
with open("overnight_results_20251223_081139/experiment.log") as f:
    first_line = f.readline()
    start_time_str = first_line.split(']')[0][1:]
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")

# Current time
now = datetime.now()
elapsed = now - start_time

# Total expected: 10 hours
total_duration = timedelta(hours=10)
remaining = total_duration - elapsed

print(f"Start time: {start_time}")
print(f"Current time: {now}")
print(f"Elapsed: {elapsed}")
print(f"Remaining: {remaining}")
print(f"Expected completion: {now + remaining}")
EOF
```

---

## Appendix: Quick Reference

### Key Files

| File | Purpose |
|------|---------|
| `run_overnight_experiments.sh` | Main experiment orchestration script |
| `sft_qwen_math.py` | Training script with full training loop |
| `math_rlvr/eval_math.py` | Evaluation script for benchmark |
| `math_rlvr/chat-fmt-base.jsonl` | Training data (4,999 examples) |
| `math_rlvr/benchmark_dataset.jsonl` | Test data (87 examples) |
| `overnight_run.log` | Master log file |
| `EXPERIMENT_STATUS.md` | Real-time status snapshot |

### Key Commands

```bash
# Monitor training
tail -f overnight_run.log

# Check GPU
nvidia-smi

# Get current step
tail -1 overnight_results_*/sft/final_model/metrics_*.jsonl | grep -oP '"step":\s*\K\d+'

# Evaluate model (after training)
cd math_rlvr && python eval_math.py --model-path ../overnight_results_*/sft/final_model

# Plot results
python visualize_sweep.py --results-dir ../overnight_results_*
```

### Expected Outcomes

| Metric | SFT | Expert Iteration | Advanced SFT |
|--------|-----|------------------|--------------|
| Training loss | 0.60-0.70 | 0.50-0.60 | 0.40-0.50 |
| Training examples | 4,999 | ~1,673 | ~3,346 |
| Benchmark accuracy | 30-35% | 40-45% | 50-55% |
| Best hyperparams | LR=5e-5 | LR=3e-5 | LR=2e-5 |

### Hypothesis Summary

1. ✓ Expert Iteration > SFT (filtering improves quality)
2. ✓ Advanced SFT > Expert Iteration (expert solutions help)
3. ✓ All methods converge (modern optimizers are robust)
4. ✓ Hyperparameters differ by method (quality affects optimal LR)

---

**End of Experiments Guide**

For questions or issues, refer to:
- OVERNIGHT_EXPERIMENT_PLAN.md (original plan)
- EXPERIMENT_STATUS.md (current status)
- SFT_README.md (training script details)
- Assignment 5 handout (theoretical background)
