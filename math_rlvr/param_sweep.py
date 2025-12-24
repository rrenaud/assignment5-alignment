#!/usr/bin/env python3
"""
Multi-objective parameter sweep for SFT training.
Optimizes for: thinking_loss, boxed_loss, and overall log-loss.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple
import numpy as np


class ParamSweepConfig:
    """Configuration for parameter sweep."""

    def __init__(self):
        # Parameter grid to search
        self.param_grid = {
            "learning_rate": [1e-5, 2e-5, 5e-5],
            "warmup_ratio": [0.05, 0.1, 0.2],
            "gradient_accumulation_steps": [2, 4, 8],
            "max_grad_norm": [0.5, 1.0, 2.0],
        }

        # Fixed parameters for benchmark
        self.fixed_params = {
            "batch_size": 4,
            "max_steps": 200,  # Fast benchmark
            "logging_steps": 10,
            "detailed_metrics_steps": 20,
            "save_steps": 10000,  # Don't save during sweep
        }

        # Data and model
        self.train_data = "benchmark_dataset.jsonl"
        self.model_name = "Qwen/Qwen2.5-Math-1.5B"

    def get_all_configs(self) -> List[Dict]:
        """Generate all parameter combinations."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        configs = []
        for combo in product(*values):
            config = dict(zip(keys, combo))
            config.update(self.fixed_params)
            configs.append(config)

        return configs


class MultiObjectiveTracker:
    """Track multiple objectives and compute Pareto frontier."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def add_result(
        self,
        config: Dict,
        thinking_loss: float,
        boxed_loss: float,
        total_loss: float,
        metrics: Dict
    ):
        """Add a result from a parameter configuration."""
        result = {
            "config": config,
            "objectives": {
                "thinking_loss": thinking_loss,
                "boxed_loss": boxed_loss,
                "total_loss": total_loss,
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)

        # Save incrementally
        self._save_results()

    def _save_results(self):
        """Save all results to JSON."""
        output_file = self.output_dir / "sweep_results.jsonl"
        with open(output_file, "w") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")

    def compute_pareto_frontier(self) -> List[Dict]:
        """
        Compute Pareto frontier for multi-objective optimization.
        A point is Pareto optimal if no other point dominates it in all objectives.

        We want to MINIMIZE all three objectives.
        """
        if not self.results:
            return []

        # Extract objective values
        objectives = np.array([
            [
                r["objectives"]["thinking_loss"],
                r["objectives"]["boxed_loss"],
                r["objectives"]["total_loss"]
            ]
            for r in self.results
        ])

        # Find Pareto frontier
        pareto_mask = self._is_pareto_efficient(objectives)
        pareto_indices = np.where(pareto_mask)[0]

        pareto_results = [self.results[i] for i in pareto_indices]

        # Save Pareto frontier
        pareto_file = self.output_dir / "pareto_frontier.jsonl"
        with open(pareto_file, "w") as f:
            for result in pareto_results:
                f.write(json.dumps(result) + "\n")

        return pareto_results

    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto efficient points.

        Args:
            costs: Array of shape (n_points, n_objectives)

        Returns:
            Boolean array indicating which points are Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Point i is dominated if there exists another point that is
                # better in all objectives
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] >= c, axis=1
                )
                is_efficient[i] = True

        return is_efficient

    def print_summary(self):
        """Print summary of sweep results."""
        if not self.results:
            print("No results yet")
            return

        print("\n" + "="*70)
        print("PARAMETER SWEEP SUMMARY")
        print("="*70)
        print(f"Total configurations tested: {len(self.results)}")

        # Best for each objective
        best_thinking = min(self.results, key=lambda r: r["objectives"]["thinking_loss"])
        best_boxed = min(self.results, key=lambda r: r["objectives"]["boxed_loss"])
        best_total = min(self.results, key=lambda r: r["objectives"]["total_loss"])

        print("\nBest for Thinking Loss:")
        self._print_result(best_thinking)

        print("\nBest for Boxed Loss:")
        self._print_result(best_boxed)

        print("\nBest for Total Loss:")
        self._print_result(best_total)

        # Compute Pareto frontier
        pareto_results = self.compute_pareto_frontier()
        print(f"\n\nPareto Frontier: {len(pareto_results)} configurations")
        print("-" * 70)
        for i, result in enumerate(pareto_results, 1):
            print(f"\n{i}. Pareto-Optimal Configuration:")
            self._print_result(result)

        print("\n" + "="*70)

    @staticmethod
    def _print_result(result: Dict):
        """Print a single result."""
        config = result["config"]
        obj = result["objectives"]

        print(f"  Objectives:")
        print(f"    Thinking Loss: {obj['thinking_loss']:.4f}")
        print(f"    Boxed Loss:    {obj['boxed_loss']:.4f}")
        print(f"    Total Loss:    {obj['total_loss']:.4f}")
        print(f"  Config:")
        print(f"    LR: {config['learning_rate']:.2e}, "
              f"Warmup: {config['warmup_ratio']:.2f}, "
              f"GradAccum: {config['gradient_accumulation_steps']}, "
              f"GradNorm: {config['max_grad_norm']:.1f}")


def extract_final_metrics(log_file: Path) -> Dict:
    """Extract final metrics from training log."""
    metrics = {
        "thinking_loss": float('inf'),
        "boxed_loss": float('inf'),
        "total_loss": float('inf'),
        "final_step": 0
    }

    if not log_file.exists():
        return metrics

    # Read metrics from the JSONL file
    metrics_file = None
    for f in log_file.parent.glob("metrics_*.jsonl"):
        metrics_file = f
        break

    if not metrics_file:
        return metrics

    # Read all metrics
    all_metrics = []
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                all_metrics.append(json.loads(line))

    if not all_metrics:
        return metrics

    # For detailed metrics (thinking/boxed), use all entries that have them
    # (they're only computed every detailed_metrics_steps)
    # For total loss, use last 20% of training
    detailed_metrics = [m for m in all_metrics if "loss/thinking_tokens" in m or "loss/boxed_tokens" in m]

    # If we have few detailed metrics, use all of them
    # Otherwise, use last half
    if len(detailed_metrics) <= 5:
        final_detailed = detailed_metrics
    else:
        final_detailed = detailed_metrics[len(detailed_metrics)//2:]

    # Average the objectives
    thinking_losses = [
        m.get("loss/thinking_tokens")
        for m in final_detailed
        if "loss/thinking_tokens" in m and m.get("stats/thinking_token_count", 0) > 0
    ]
    boxed_losses = [
        m.get("loss/boxed_tokens")
        for m in final_detailed
        if "loss/boxed_tokens" in m and m.get("stats/boxed_token_count", 0) > 0
    ]

    # For total loss, get all metrics with loss/train
    metrics_with_loss = [m for m in all_metrics if "loss/train" in m or "loss/total" in m]

    # Use last 50% of those
    if len(metrics_with_loss) > 2:
        final_metrics_total = metrics_with_loss[len(metrics_with_loss)//2:]
    else:
        final_metrics_total = metrics_with_loss

    # Extract total losses
    total_losses = []
    for m in final_metrics_total:
        if "loss/train" in m:
            total_losses.append(m["loss/train"])
        elif "loss/total" in m:
            total_losses.append(m["loss/total"])

    if thinking_losses:
        metrics["thinking_loss"] = float(np.mean(thinking_losses))
    if boxed_losses:
        metrics["boxed_loss"] = float(np.mean(boxed_losses))
    if total_losses:
        metrics["total_loss"] = float(np.mean(total_losses))

    metrics["final_step"] = all_metrics[-1].get("step", 0)

    return metrics


def run_single_config(
    config: Dict,
    run_id: int,
    sweep_dir: Path,
    train_script: str = "sft_qwen_math.py"
) -> Dict:
    """Run training with a single configuration."""
    output_dir = sweep_dir / f"run_{run_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python3",
        train_script,
        "--train-data", config.get("train_data", "benchmark_dataset.jsonl"),
        "--output-dir", str(output_dir),
        "--learning-rate", str(config["learning_rate"]),
        "--warmup-ratio", str(config["warmup_ratio"]),
        "--gradient-accumulation-steps", str(config["gradient_accumulation_steps"]),
        "--max-grad-norm", str(config["max_grad_norm"]),
        "--batch-size", str(config["batch_size"]),
        "--max-steps", str(config["max_steps"]),
        "--logging-steps", str(config["logging_steps"]),
        "--detailed-metrics-steps", str(config["detailed_metrics_steps"]),
        "--save-steps", str(config["save_steps"]),
        "--bf16",
        "--disable-tensorboard",  # Faster without TB
    ]

    print(f"\n{'='*70}")
    print(f"Running Configuration {run_id}")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd[:10])}...")

    # Run training
    log_file = output_dir / "training.log"
    start_time = time.time()

    try:
        with open(log_file, "w") as log:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800  # 30 min timeout
            )
        success = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        print(f"⚠ Run {run_id} timed out")
        success = False
    except Exception as e:
        print(f"✗ Run {run_id} failed: {e}")
        success = False

    elapsed_time = time.time() - start_time

    # Extract metrics
    metrics = extract_final_metrics(log_file)
    metrics["success"] = success
    metrics["elapsed_time"] = elapsed_time

    print(f"\nResults for Run {run_id}:")
    print(f"  Status: {'✓ Success' if success else '✗ Failed'}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"  Thinking Loss: {metrics['thinking_loss']:.4f}")
    print(f"  Boxed Loss: {metrics['boxed_loss']:.4f}")
    print(f"  Total Loss: {metrics['total_loss']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-objective parameter sweep")
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="./param_sweep_results",
        help="Directory to store sweep results"
    )
    parser.add_argument(
        "--benchmark-data",
        type=str,
        default="benchmark_dataset.jsonl",
        help="Benchmark dataset to use"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to test (None = all)"
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="sft_qwen_math.py",
        help="Path to training script"
    )

    args = parser.parse_args()

    # Setup
    sweep_dir = Path(args.sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Initialize sweep config
    sweep_config = ParamSweepConfig()
    sweep_config.fixed_params["train_data"] = args.benchmark_data

    # Get all configurations
    all_configs = sweep_config.get_all_configs()
    print(f"\n{'='*70}")
    print(f"MULTI-OBJECTIVE PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Total configurations: {len(all_configs)}")
    print(f"Parameter grid:")
    for param, values in sweep_config.param_grid.items():
        print(f"  {param}: {values}")

    # Limit configs if requested
    if args.max_configs and args.max_configs < len(all_configs):
        import random
        random.shuffle(all_configs)
        all_configs = all_configs[:args.max_configs]
        print(f"\nLimited to {args.max_configs} random configurations")

    # Initialize tracker
    tracker = MultiObjectiveTracker(sweep_dir)

    # Run sweep
    print(f"\nStarting sweep...")
    start_time = time.time()

    for i, config in enumerate(all_configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Configuration {i}/{len(all_configs)}")
        print(f"{'#'*70}")

        metrics = run_single_config(
            config,
            run_id=i,
            sweep_dir=sweep_dir,
            train_script=args.train_script
        )

        if metrics["success"]:
            tracker.add_result(
                config=config,
                thinking_loss=metrics["thinking_loss"],
                boxed_loss=metrics["boxed_loss"],
                total_loss=metrics["total_loss"],
                metrics=metrics
            )

    total_time = time.time() - start_time

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful runs: {len(tracker.results)}/{len(all_configs)}")

    tracker.print_summary()

    # Save final summary
    summary_file = sweep_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Parameter Sweep Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Total configurations: {len(all_configs)}\n")
        f.write(f"Successful runs: {len(tracker.results)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"\nResults saved to: {sweep_dir}\n")

    print(f"\nResults saved to: {sweep_dir}")
    print(f"  - sweep_results.jsonl: All results")
    print(f"  - pareto_frontier.jsonl: Pareto-optimal configurations")
    print(f"  - summary.txt: Summary statistics")


if __name__ == "__main__":
    main()
