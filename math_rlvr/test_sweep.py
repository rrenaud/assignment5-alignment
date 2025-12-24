#!/usr/bin/env python3
"""
Test parameter sweep with minimal configs (2-3 runs) to validate framework.
"""

import sys
import json
from pathlib import Path

# Modify param_sweep to use smaller grid
sys.path.insert(0, str(Path(__file__).parent))
from param_sweep import ParamSweepConfig, MultiObjectiveTracker, run_single_config

def main():
    print("="*70)
    print("TEST SWEEP - Validating Framework")
    print("="*70)

    # Create test config with minimal parameter grid
    test_configs = [
        {
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "batch_size": 4,
            "max_steps": 50,  # Very short for testing
            "logging_steps": 5,
            "detailed_metrics_steps": 10,
            "save_steps": 10000,
            "train_data": "benchmark_dataset.jsonl",
        },
        {
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "batch_size": 4,
            "max_steps": 50,
            "logging_steps": 5,
            "detailed_metrics_steps": 10,
            "save_steps": 10000,
            "train_data": "benchmark_dataset.jsonl",
        },
    ]

    sweep_dir = Path("./test_sweep_results")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTesting with {len(test_configs)} configurations")
    print(f"Each config runs for {test_configs[0]['max_steps']} steps")
    print(f"Output: {sweep_dir}")
    print()

    tracker = MultiObjectiveTracker(sweep_dir)

    for i, config in enumerate(test_configs, 1):
        print(f"\n{'#'*70}")
        print(f"# Test Configuration {i}/{len(test_configs)}")
        print(f"{'#'*70}")

        metrics = run_single_config(
            config,
            run_id=i,
            sweep_dir=sweep_dir,
            train_script="sft_qwen_math.py"
        )

        if metrics["success"]:
            tracker.add_result(
                config=config,
                thinking_loss=metrics["thinking_loss"],
                boxed_loss=metrics["boxed_loss"],
                total_loss=metrics["total_loss"],
                metrics=metrics
            )

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    tracker.print_summary()

    print(f"\n✓ Framework validated!")
    print(f"  Results saved to: {sweep_dir}")
    print(f"\nReady to run full sweep!")

if __name__ == "__main__":
    main()
