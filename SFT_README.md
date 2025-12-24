# SFT Training for Qwen-2.5-Math-1.5B

Custom supervised fine-tuning script without using HuggingFace Trainer.

## Features

- ✅ **Custom PyTorch training loop** - Full control over training process
- ✅ **Gradient accumulation** - Train with larger effective batch sizes
- ✅ **Checkpoint saving/resuming** - Save at regular intervals and resume training
- ✅ **Chat format training** - Uses tokenizer's chat template
- ✅ **Label masking** - Only compute loss on assistant responses (not user prompts)
- ✅ **Mixed precision** - Support for bf16/fp16 training
- ✅ **Cosine learning rate schedule** - With warmup
- ✅ **Gradient clipping** - Prevent exploding gradients
- ✅ **Progress tracking** - Real-time loss and learning rate monitoring

## Data Format

The script expects JSONL files with this structure (matches `deepinfra_qwen3_32b_math.jsonl`):

```json
{
  "problem": "How many vertical asymptotes...",
  "ground_truth": "The denominator factors...",
  "predicted": "Thus, the graph has 2 asymptotes",
  "correct": true,
  "level": "Level 3",
  "subject": "Algebra",
  "response": "<think>...</think> <answer>...</answer>"
}
```

## Quick Start

### Basic Training

```bash
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_output \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --bf16
```

### All Arguments

```
Data Arguments:
  --train-data PATH              Path to training JSONL file
  --filter-correct               Only use correct responses
  --include-ground-truth         Also train on ground truth solutions

Model Arguments:
  --model-name NAME              Model name/path (default: Qwen/Qwen2.5-Math-1.5B)
  --max-length INT               Max sequence length (default: 2048)

Training Arguments:
  --output-dir PATH              Output directory for checkpoints
  --num-epochs INT               Number of epochs (default: 3)
  --batch-size INT               Batch size per device (default: 4)
  --gradient-accumulation-steps  Gradient accumulation (default: 4)
  --learning-rate FLOAT          Learning rate (default: 2e-5)
  --weight-decay FLOAT           Weight decay (default: 0.01)
  --max-grad-norm FLOAT          Gradient clipping (default: 1.0)
  --warmup-ratio FLOAT           Warmup ratio (default: 0.1)
  --save-steps INT               Save every N steps (default: 1000)
  --logging-steps INT            Log every N steps (default: 10)
  --resume-from PATH             Resume from checkpoint

System Arguments:
  --num-workers INT              Dataloader workers (default: 4)
  --bf16                         Use bfloat16 precision
  --fp16                         Use float16 precision
```

## Training Strategies

### 1. Train on All Responses (Default)

Trains on all 8 responses per problem, including varied reasoning paths:

```bash
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --output-dir ./qwen_math_sft_all
```

**Pros:** More diverse reasoning patterns, larger dataset
**Cons:** May include incorrect solutions

### 2. Train Only on Correct Responses

If responses have been verified, train only on correct ones:

```bash
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --filter-correct \
    --output-dir ./qwen_math_sft_correct
```

**Pros:** Higher quality data
**Cons:** Smaller dataset, requires verification step

### 3. Include Ground Truth Solutions

Augment training with original MATH dataset solutions:

```bash
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --include-ground-truth \
    --output-dir ./qwen_math_sft_with_gt
```

**Pros:** Balanced dataset with known-good solutions
**Cons:** Ground truth may be more concise than model responses

## GPU Memory Requirements

**Qwen-2.5-Math-1.5B** (1.5 billion parameters)

| Precision | Batch Size | Grad Accum | Memory Required |
|-----------|------------|------------|-----------------|
| bf16      | 4          | 4          | ~12 GB          |
| bf16      | 8          | 2          | ~16 GB          |
| fp16      | 4          | 4          | ~12 GB          |
| fp32      | 2          | 8          | ~20 GB          |

*Estimates for max_length=2048. Actual usage depends on sequence length distribution.*

## Checkpoint Structure

```
qwen_math_sft_output/
├── checkpoint-epoch0-step1000/
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   └── training_state.pt       # optimizer, scheduler, step info
├── checkpoint-epoch1-step2000/
│   └── ...
└── final_model/
    ├── config.json
    ├── model.safetensors
    └── generation_config.json
```

## Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
python3 sft_qwen_math.py \
    --train-data deepinfra_qwen3_32b_math.jsonl \
    --resume-from ./qwen_math_sft_output/checkpoint-epoch1-step2000 \
    --output-dir ./qwen_math_sft_output \
    --num-epochs 3
```

## Monitoring Training

The script outputs:
- Real-time progress bar with loss and learning rate
- Average loss per epoch
- Checkpoint saves at regular intervals

Example output:
```
Training: 45%|████▌     | 4500/10000 [1:23:45<1:31:12, loss=0.8234, lr=1.5e-05, epoch=2]
```

## Using the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./qwen_math_sft_output/final_model")
tokenizer = AutoTokenizer.from_pretrained("./qwen_math_sft_output")

messages = [{"role": "user", "content": "What is 2+2?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Tips

1. **Start with lower learning rate** (1e-5) if training is unstable
2. **Use bf16** if your GPU supports it (A100, H100, RTX 40-series)
3. **Increase gradient accumulation** if you run out of memory
4. **Save frequently** (--save-steps 500) for long training runs
5. **Monitor loss curve** - should decrease steadily, not spike
6. **Validate periodically** - test model on held-out problems

## Expected Training Time

On a single A100 GPU (40GB):
- ~40,000 samples (8 responses × 5000 problems)
- Batch size 4, gradient accumulation 4
- 3 epochs
- **Estimated time: 8-12 hours**

## Common Issues

**Out of Memory:**
- Reduce `--batch-size` to 2 or 1
- Increase `--gradient-accumulation-steps` to maintain effective batch size
- Reduce `--max-length` to 1024 or 1536

**Loss not decreasing:**
- Lower learning rate (try 1e-5 or 5e-6)
- Increase warmup ratio (try 0.2)
- Check data quality

**Training too slow:**
- Increase `--num-workers` for faster data loading
- Reduce sequence length if padding is excessive
- Use multiple GPUs (requires modifying script for DDP)
