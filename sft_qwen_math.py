#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for Qwen2.5-Math-1.5B

This script implements custom SFT training without using HuggingFace Trainer.
Features:
- Custom PyTorch training loop with full control
- Gradient accumulation for larger effective batch sizes
- Checkpoint saving and resuming
- Chat format training with tokenizer's chat template
- Label masking (loss only on assistant responses)
- Mixed precision (bf16/fp16) support
- Cosine LR schedule with warmup
- Gradient clipping
- Comprehensive metrics logging with TensorBoard and W&B support
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

# TensorBoard support (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# Weights & Biases support (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Install with: pip install wandb")


# ============================================================================
# Core SFT Functions (imported by tests/adapters.py)
# ============================================================================

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)

    # Tokenize prompts and outputs SEPARATELY to avoid token merging at boundary
    prompt_encodings = tokenizer(prompt_strs, add_special_tokens=True, padding=False)
    output_encodings = tokenizer(output_strs, add_special_tokens=False, padding=False)

    # Concatenate token IDs
    concatenated_ids = []
    for i in range(batch_size):
        prompt_ids = prompt_encodings["input_ids"][i]
        output_ids = output_encodings["input_ids"][i]
        concatenated_ids.append(prompt_ids + output_ids)

    # Pad concatenated sequences
    max_len = max(len(ids) for ids in concatenated_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Pad to max_len (slicing will remove last token to create input_ids and labels)
    padded_ids = []
    attention_masks = []
    for ids in concatenated_ids:
        seq_len = len(ids)
        padding_len = max_len - seq_len
        padded_ids.append(ids + [pad_token_id] * padding_len)
        attention_masks.append([1] * seq_len + [0] * padding_len)

    input_ids_full = torch.tensor(padded_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)

    # Create response mask: 1 for response tokens, 0 for prompt/padding
    response_mask = torch.zeros_like(input_ids_full, dtype=torch.float32)

    for i in range(batch_size):
        prompt_len = len(prompt_encodings["input_ids"][i])
        seq_len = len(concatenated_ids[i])
        # Mark response tokens (after prompt, before padding)
        response_mask[i, prompt_len:seq_len] = 1.0

    # Slice off last token for causal LM (input predicts next token)
    input_ids = input_ids_full[:, :-1].contiguous()
    labels = input_ids_full[:, 1:].contiguous()  # Shifted by 1
    response_mask = response_mask[:, 1:].contiguous()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits.
    
    Args:
        logits: torch.Tensor of shape (..., vocab_size)
    
    Returns:
        torch.Tensor of shape (...): entropy values
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get log probabilities of the response tokens.
    
    Args:
        model: the model
        input_ids: shape (batch_size, seq_len)
        labels: shape (batch_size, seq_len)
        return_token_entropy: whether to return per-token entropy
    
    Returns:
        dict with:
            "log_probs": shape (batch_size, seq_len)
            "token_entropy": shape (batch_size, seq_len) if return_token_entropy else None
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    # Get log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Gather log probs of actual tokens (labels)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    result = {"log_probs": token_log_probs}
    
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    
    return result


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.
    
    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.
    
    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    # Convert mask to float for multiplication
    mask_float = mask.float()
    
    # Zero out masked elements
    masked_tensor = tensor * mask_float
    
    # Sum over the specified dimension
    if dim is None:
        # Sum over all dimensions and divide by total count of non-masked elements
        total = masked_tensor.sum()
        count = mask_float.sum()
    else:
        # Sum along specified dimension
        total = masked_tensor.sum(dim=dim)
        count = mask_float.sum(dim=dim)
    
    # Allow NaN when count=0 (don't clamp)
    return total / count


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the sum of masked elements divided by normalize_constant.
    """
    # Convert mask to float
    mask_float = mask.float()

    # Zero out masked elements
    masked_tensor = tensor * mask_float

    # Sum over the specified dimension
    if dim is None:
        total = masked_tensor.sum()
    else:
        total = masked_tensor.sum(dim=dim)

    # Normalize by constant
    return total / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the SFT loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: shape (batch_size, seq_len)
        response_mask: shape (batch_size, seq_len)
        gradient_accumulation_steps: number of gradient accumulation steps
        normalize_constant: normalization constant for the loss

    Returns:
        (loss, metadata_dict)
    """
    # Negative log likelihood loss (only on response tokens)
    nll_loss = -policy_log_probs

    # Compute loss per batch element (average over sequence)
    # Then average over batch
    batch_size = policy_log_probs.size(0)

    # Sum over masked elements and normalize
    total_loss = masked_normalize(
        nll_loss,
        response_mask,
        dim=None,
        normalize_constant=normalize_constant,
    )

    # Average over batch
    loss = total_loss / batch_size

    # Scale by gradient accumulation before backward
    scaled_loss = loss / gradient_accumulation_steps

    # Backward pass
    scaled_loss.backward()

    return scaled_loss, {}


# ============================================================================
# Helper Functions
# ============================================================================

def find_boxed_tokens(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Find tokens that are inside \\boxed{...} markup.
    
    Returns a boolean mask of shape (batch_size, seq_len).
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for i in range(batch_size):
        text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
        
        # Find all \boxed{...} regions
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = list(re.finditer(boxed_pattern, text))
        
        if matches:
            # Mark tokens in boxed regions
            for match in matches:
                start_char, end_char = match.span()
                # Convert char positions to token positions (approximate)
                tokens_before_match = tokenizer.encode(text[:start_char], add_special_tokens=False)
                tokens_in_match = tokenizer.encode(match.group(1), add_special_tokens=False)
                
                start_token = len(tokens_before_match)
                end_token = start_token + len(tokens_in_match)
                
                if end_token <= seq_len:
                    mask[i, start_token:end_token] = True
    
    return mask


def find_thinking_tokens(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Find tokens that are inside <think>...</think> markup.
    
    Returns a boolean mask of shape (batch_size, seq_len).
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for i in range(batch_size):
        text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
        
        # Find all <think>...</think> regions
        thinking_pattern = r'<think>(.*?)</think>'
        matches = list(re.finditer(thinking_pattern, text, re.DOTALL))
        
        if matches:
            for match in matches:
                start_char, end_char = match.span()
                tokens_before_match = tokenizer.encode(text[:start_char], add_special_tokens=False)
                tokens_in_match = tokenizer.encode(match.group(1), add_special_tokens=False)
                
                start_token = len(tokens_before_match)
                end_token = start_token + len(tokens_in_match)
                
                if end_token <= seq_len:
                    mask[i, start_token:end_token] = True
    
    return mask


# ============================================================================
# Metrics Logger
# ============================================================================

class MetricsLogger:
    """Comprehensive metrics tracking for training."""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # JSON log file for detailed metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_path = self.log_dir / f"metrics_{timestamp}.jsonl"

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            print(f"✓ TensorBoard logging to: {self.log_dir / 'tensorboard'}")
        
        # Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project or "qwen-math-sft",
                name=wandb_name,
                config=wandb_config or {},
                dir=str(self.log_dir),
            )
            print(f"✓ W&B logging to project: {wandb_project or 'qwen-math-sft'}")

        # Running statistics
        self.step_metrics = []

    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics to JSON, TensorBoard, and W&B."""
        # Add timestamp and step
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()

        # Write to JSON log
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Write to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ["step", "epoch"]:
                    self.writer.add_scalar(key, value, step)
        
        # Write to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def close(self):
        """Close all logging backends."""
        if self.writer:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()


# ============================================================================
# Dataset
# ============================================================================

class MathSFTDataset(Dataset):
    """Dataset for math SFT training from JSONL files."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        filter_correct: bool = False,
        include_ground_truth: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.filter_correct = filter_correct
        self.include_ground_truth = include_ground_truth

        # Load data
        self.examples = []
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                
                # Filter if requested
                if filter_correct and not item.get("correct", True):
                    continue
                
                # Add model-generated response
                prompt = item.get("problem", "")
                response = item.get("response", "")
                
                if prompt and response:
                    self.examples.append({"prompt": prompt, "output": response})
                
                # Add ground truth if requested
                if include_ground_truth and "ground_truth" in item:
                    self.examples.append({
                        "prompt": prompt,
                        "output": item["ground_truth"]
                    })

        print(f"Loaded {len(self.examples)} training examples from {data_path}")
        if filter_correct:
            print("  (filtered to correct responses only)")
        if include_ground_truth:
            print("  (includes ground truth solutions)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as chat messages
        messages = [{"role": "user", "content": example["prompt"]}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Full text includes response
        full_text = prompt_text + example["output"]
        
        return {
            "prompt": prompt_text,
            "output": example["output"],
            "full_text": full_text,
        }


def collate_fn(batch, tokenizer, max_length=2048):
    """Collate function for DataLoader."""
    prompts = [item["prompt"] for item in batch]
    outputs = [item["output"] for item in batch]
    
    # Tokenize with response masking
    tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
    
    # Truncate if needed
    max_len = min(tokenized["input_ids"].size(1), max_length)
    
    return {
        "input_ids": tokenized["input_ids"][:, :max_len],
        "labels": tokenized["labels"][:, :max_len],
        "response_mask": tokenized["response_mask"][:, :max_len],
    }


# ============================================================================
# Training Function
# ============================================================================

def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    args: argparse.Namespace,
):
    """Main training loop."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, args.max_length),
    )
    
    # Calculate total steps
    if args.max_steps:
        total_steps = args.max_steps
        num_epochs = (total_steps // len(train_loader)) + 1
    else:
        num_epochs = args.num_epochs
        total_steps = len(train_loader) * num_epochs
    
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Metrics logger
    metrics_logger = MetricsLogger(
        log_dir=args.output_dir,
        use_tensorboard=True,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_config=vars(args),
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if args.fp16 else None
    autocast_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    
    # Training loop
    model.train()
    global_step = 0
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        checkpoint_path = Path(args.resume_from) / "training_state.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["step"]
            print(f"Resumed from checkpoint: epoch {start_epoch}, step {global_step}")
    
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"  Model: {args.model_name}")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Precision: {'bfloat16' if args.bf16 else 'float16' if args.fp16 else 'float32'}")
    print("=" * 60 + "\n")
    
    pbar = tqdm(total=total_steps, initial=global_step, desc="Training")
    
    for epoch in range(start_epoch, num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits
                
                # Get log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # Compute loss using our SFT function
                token_log_probs.requires_grad = True
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=token_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    normalize_constant=1.0,
                )
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                # loss.backward() already called in sft_microbatch_train_step
                pass
            
            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % args.logging_steps == 0:
                    metrics = {
                        "loss/train": loss.item() * args.gradient_accumulation_steps,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "gradient_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "epoch": epoch + 1,
                    }
                    
                    # Detailed metrics every N steps
                    if global_step % (args.logging_steps * 10) == 0:
                        with torch.no_grad():
                            # Compute detailed losses
                            nll_per_token = -token_log_probs
                            total_loss = masked_mean(nll_per_token, response_mask, dim=None)
                            
                            # Token accuracy
                            predicted_tokens = logits.argmax(dim=-1)
                            correct = (predicted_tokens == labels).float()
                            token_accuracy = masked_mean(correct, response_mask, dim=None)
                            
                            # Perplexity
                            perplexity = torch.exp(total_loss)
                            
                            # Boxed tokens analysis
                            boxed_mask = find_boxed_tokens(input_ids, tokenizer).float().to(device)
                            boxed_response_mask = boxed_mask * response_mask
                            
                            thinking_mask = find_thinking_tokens(input_ids, tokenizer).float().to(device)
                            thinking_response_mask = thinking_mask * response_mask
                            
                            # Other tokens (not boxed, not thinking)
                            other_mask = response_mask - boxed_response_mask - thinking_response_mask
                            other_mask = torch.clamp(other_mask, min=0.0)
                            
                            boxed_count = boxed_response_mask.sum()
                            thinking_count = thinking_response_mask.sum()
                            other_count = other_mask.sum()
                            total_count = response_mask.sum()
                            
                            metrics.update({
                                "loss/total": total_loss.item(),
                                "loss/perplexity": perplexity.item(),
                                "accuracy/token": token_accuracy.item(),
                                "loss/per_token_avg": total_loss.item(),
                                "loss/boxed_tokens": masked_mean(nll_per_token, boxed_response_mask, dim=None).item() if boxed_count > 0 else 0.0,
                                "stats/boxed_token_count": boxed_count.item(),
                                "accuracy/boxed_tokens": masked_mean(correct, boxed_response_mask, dim=None).item() if boxed_count > 0 else 0.0,
                                "loss/thinking_tokens": masked_mean(nll_per_token, thinking_response_mask, dim=None).item() if thinking_count > 0 else 0.0,
                                "stats/thinking_token_count": thinking_count.item(),
                                "loss/other_tokens": masked_mean(nll_per_token, other_mask, dim=None).item() if other_count > 0 else 0.0,
                                "stats/other_token_count": other_count.item(),
                                "stats/total_tokens": total_count.item(),
                                "stats/batch_size": float(args.batch_size),
                                "stats/boxed_pct": (boxed_count / total_count * 100).item() if total_count > 0 else 0.0,
                                "stats/thinking_pct": (thinking_count / total_count * 100).item() if total_count > 0 else 0.0,
                                "stats/other_pct": (other_count / total_count * 100).item() if total_count > 0 else 0.0,
                            })
                    
                    metrics_logger.log_metrics(metrics, global_step)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{metrics['loss/train']:.4f}",
                        "lr": f"{metrics['learning_rate']:.2e}",
                        "grad": f"{metrics['gradient_norm']:.2f}",
                        "epoch": epoch + 1,
                    })
                    if "loss/boxed_tokens" in metrics:
                        pbar.set_postfix({
                            **pbar.postfix,
                            "boxed": f"{metrics['loss/boxed_tokens']:.3f}",
                        })
                
                pbar.update(1)
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-epoch{epoch}-step{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save model
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save training state
                    torch.save({
                        "epoch": epoch,
                        "step": global_step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, checkpoint_dir / "training_state.pt")
                    
                    print(f"\nCheckpoint saved to {checkpoint_dir}")
    
    pbar.close()
    
    # Save final model
    final_dir = Path(args.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to {final_dir}")
    
    # Close loggers
    metrics_logger.close()
    
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT training for Qwen2.5-Math")
    
    # Data arguments
    parser.add_argument("--train-data", type=str, required=True,
                       help="Path to training JSONL file")
    parser.add_argument("--filter-correct", action="store_true",
                       help="Only use correct responses")
    parser.add_argument("--include-ground-truth", action="store_true",
                       help="Also train on ground truth solutions")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                       help="Model name or path")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum training steps (overrides num_epochs)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--save-steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Log metrics every N steps")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # System arguments
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 precision")
    
    # W&B arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="qwen-math-sft",
                       help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                       help="W&B run name")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset
    train_dataset = MathSFTDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        filter_correct=args.filter_correct,
        include_ground_truth=args.include_ground_truth,
    )
    
    # Train
    train(model, tokenizer, train_dataset, args)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
