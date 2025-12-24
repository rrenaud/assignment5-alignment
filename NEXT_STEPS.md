# Assignment 5: Next Steps

## ✅ Completed: Supervised Fine-Tuning (SFT)

**Status**: All 10 SFT tests passing!

**What you accomplished**:
- ✅ `tokenize_prompt_and_output()` - Tokenization with response masking
- ✅ `compute_entropy()` - Entropy calculation
- ✅ `get_response_log_probs()` - Log probability computation
- ✅ `masked_mean()` - Masked averaging
- ✅ `masked_normalize()` - Masked normalization
- ✅ `sft_microbatch_train_step()` - SFT training step

**Bonus achievements**:
- ✅ Refactored core logic to `sft_qwen_math.py`
- ✅ Completed hyperparameter sweep identifying optimal configs
- ✅ Sanity checked training script

---

## 🎯 Next: GRPO (Group Relative Policy Optimization)

**Test file**: `tests/test_grpo.py`

### Functions to Implement

#### 1. **`run_compute_group_normalized_rewards()`**
```python
def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
```

**Purpose**: Compute rewards for rollout responses and normalize them by group
- Takes multiple rollout responses per prompt (group_size responses per problem)
- Computes rewards using a reward function
- Normalizes within each group (subtract mean, optionally divide by std)
- Returns normalized rewards (advantages), raw rewards, and metadata

**Key concept**: GRPO normalizes rewards within groups to reduce variance

**Tests**:
- `test_compute_group_normalized_rewards_normalize_by_std`
- `test_compute_group_normalized_rewards_no_normalize_by_std`

---

#### 2. **`run_compute_naive_policy_gradient_loss()`**
```python
def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
```

**Purpose**: Compute basic REINFORCE policy gradient loss
- Formula: `loss = -rewards * log_probs` (per token)
- Used as baseline for policy gradient methods
- Can use either raw rewards or advantages

**Test**: `test_compute_naive_policy_gradient_loss`

---

#### 3. **`run_compute_grpo_clip_loss()`**
```python
def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

**Purpose**: Compute PPO-style clipped policy gradient loss for GRPO
- Computes ratio between new and old policy probabilities
- Clips ratio to prevent large policy updates
- Uses advantages (group-normalized rewards)
- Returns loss and metadata (for clip fraction tracking)

**Formula** (per token):
```
ratio = exp(new_log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1-cliprange, 1+cliprange)
loss = -min(advantages * ratio, advantages * clipped_ratio)
```

**Tests**:
- `test_compute_grpo_clip_loss_large_cliprange`
- `test_compute_grpo_clip_loss_small_cliprange`

---

#### 4. **`run_compute_policy_gradient_loss()`**
```python
def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

**Purpose**: Wrapper that dispatches to the correct loss function
- Supports three loss types:
  - `"no_baseline"`: Uses raw rewards with naive loss
  - `"reinforce_with_baseline"`: Uses advantages with naive loss
  - `"grpo_clip"`: Uses PPO-style clipping with advantages

**Tests**:
- `test_compute_policy_gradient_loss_no_baseline`
- `test_compute_policy_gradient_loss_reinforce_with_baseline`
- `test_compute_policy_gradient_loss_grpo_clip`

---

#### 5. **`run_grpo_microbatch_train_step()`**
```python
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

**Purpose**: Complete GRPO training step
- Computes policy gradient loss (using one of the loss types)
- Applies response masking
- Performs backward pass with gradient accumulation
- Returns loss and metrics

**Tests**:
- `test_grpo_microbatch_train_step_grpo_clip`
- `test_grpo_microbatch_train_step_grpo_clip_10_steps`

---

### Implementation Order (Recommended)

1. **Start with `run_compute_naive_policy_gradient_loss()`** (simplest)
   - Pure REINFORCE: `loss = -rewards * log_probs`

2. **Then `run_compute_group_normalized_rewards()`**
   - Reshape rollout_responses into groups
   - Compute rewards for each response
   - Normalize within groups: `(reward - group_mean) / (group_std + eps)`

3. **Then `run_compute_grpo_clip_loss()`**
   - Compute probability ratios
   - Apply PPO-style clipping
   - Return clipped loss

4. **Then `run_compute_policy_gradient_loss()`** (wrapper)
   - Dispatch to appropriate loss function based on loss_type

5. **Finally `run_grpo_microbatch_train_step()`**
   - Use the loss functions you've built
   - Apply masking with `masked_mean()` (already implemented)
   - Handle gradient accumulation

---

## 🔍 Key Concepts to Understand

### GRPO vs SFT
- **SFT**: Maximize likelihood of human-written responses
- **GRPO**: Maximize rewards through policy gradients with group normalization

### Group Normalization
- Each prompt gets `group_size` rollout responses
- Rewards are normalized within each group
- Reduces variance and improves training stability

### PPO-style Clipping
- Prevents large policy updates
- Clips probability ratios to `[1-ε, 1+ε]`
- Improves stability over vanilla policy gradients

---

## 📚 Reference Materials

**Papers**:
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- PPO (for clipping): https://arxiv.org/abs/1707.06347

**Existing Code**:
- `sft_qwen_math.py` - Your working SFT implementation
- `cs336_alignment/drgrpo_grader.py` - Reward verification system

---

## 🧪 Testing Strategy

1. **Run individual tests** as you implement:
   ```bash
   uv run pytest tests/test_grpo.py::test_compute_naive_policy_gradient_loss -v
   ```

2. **Run all GRPO tests** when done:
   ```bash
   uv run pytest tests/test_grpo.py -v
   ```

3. **Use snapshots** to verify correctness:
   - Tests compare against pre-computed snapshots in `tests/_snapshots/`
   - Snapshots ensure exact numerical correctness

---

## 📊 After GRPO: Optional Assignments

Once GRPO is complete, you can optionally work on:

### **DPO (Direct Preference Optimization)**
- Test file: `tests/test_dpo.py`
- Function: `run_compute_per_instance_dpo_loss()`
- Simpler alternative to RLHF

### **Safety & Instruction Tuning**
- Data utilities: `get_packed_sft_dataset()`, `run_iterate_batches()`
- Response parsing: `run_parse_mmlu_response()`, `run_parse_gsm8k_response()`
- See: `cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`

---

## 🎯 Your Current Position

```
Assignment Progress:

✅ SFT Implementation        [====================] 100%
   ✅ All core functions
   ✅ Hyperparameter sweep
   ✅ Sanity check passed

🎯 GRPO Implementation       [                    ] 0%
   ⏳ 5 functions to implement
   ⏳ ~10 tests to pass

⏸️  Expert Iteration         [                    ] Not started
   (May be part of experimental section)

⏸️  DPO (Optional)           [                    ] Not started

⏸️  Safety/RLHF (Optional)   [                    ] Not started
```

---

## 🚀 Recommended Next Action

**Start implementing GRPO functions in this order**:

```bash
# 1. Implement the simplest function first
#    Location: tests/adapters.py (lines 227-243)
vim tests/adapters.py  # or your editor of choice

# 2. Test as you go
uv run pytest tests/test_grpo.py::test_compute_naive_policy_gradient_loss -v

# 3. Move to the next function and repeat
```

**Pro tip**: Similar to SFT, you should:
1. Implement the functions in `tests/adapters.py` first
2. Then consider refactoring to a separate module if needed
3. Run tests frequently to catch issues early

Good luck with GRPO! 🚀
