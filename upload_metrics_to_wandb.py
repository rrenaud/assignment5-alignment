#!/usr/bin/env python3
"""
Upload existing SFT hparam search metrics to Weights & Biases.
Idempotent: safe to run multiple times without duplicating data.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

import wandb


def get_run_id(search_dir: str, run_name: str) -> str:
    """Generate a deterministic run ID based on search dir and run name."""
    unique_str = f"{search_dir}:{run_name}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:8]


def upload_run_metrics(
    run_dir: Path,
    project: str,
    entity: Optional[str] = None,
    search_name: str = "",
) -> bool:
    """Upload metrics from a single run directory to W&B.

    Returns True if successful, False otherwise.
    """
    # Load config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        print(f"  Skipping {run_dir.name}: no config.json")
        return False

    with open(config_path) as f:
        config = json.load(f)

    # Find metrics file
    metrics_files = list(run_dir.glob("metrics_*.jsonl"))
    if not metrics_files:
        print(f"  Skipping {run_dir.name}: no metrics file")
        return False

    metrics_file = metrics_files[0]

    # Load all metrics
    metrics_list = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metrics_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not metrics_list:
        print(f"  Skipping {run_dir.name}: no metrics data")
        return False

    # Generate deterministic run ID for idempotency
    run_id = get_run_id(str(run_dir.parent), run_dir.name)
    run_name = f"{search_name}_{run_dir.name}" if search_name else run_dir.name

    # Check if run already exists and has all data
    api = wandb.Api()
    try:
        existing_run = api.run(f"{entity or api.default_entity}/{project}/{run_id}")
        existing_step = existing_run.summary.get("step", 0)
        max_step = max(m.get("step", 0) for m in metrics_list)

        if existing_step >= max_step:
            print(f"  {run_dir.name}: already uploaded (step {existing_step}), skipping")
            return True
        else:
            print(f"  {run_dir.name}: resuming from step {existing_step} to {max_step}")
            resume = "must"
    except wandb.errors.CommError:
        # Run doesn't exist yet
        print(f"  {run_dir.name}: creating new run")
        resume = "allow"

    # Initialize W&B run
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        id=run_id,
        config=config,
        resume=resume,
    )

    # Get last logged step to avoid duplicates
    try:
        last_step = run.summary.get("step", -1)
    except:
        last_step = -1

    # Upload metrics
    logged_count = 0
    for metrics in metrics_list:
        step = metrics.get("step", 0)
        if step <= last_step:
            continue  # Skip already logged steps

        # Remove timestamp from metrics (not needed in W&B)
        log_metrics = {k: v for k, v in metrics.items() if k != "timestamp"}
        wandb.log(log_metrics, step=step)
        logged_count += 1

    # Log final summary
    final_metrics = metrics_list[-1]
    for key in ["loss/val", "loss/train", "accuracy/val_token"]:
        if key in final_metrics:
            run.summary[key] = final_metrics[key]

    wandb.finish()
    print(f"  {run_dir.name}: uploaded {logged_count} new metric entries")
    return True


def upload_search_results(
    search_dir: str,
    project: str = "sft-hparam-search",
    entity: Optional[str] = None,
):
    """Upload all runs from a hyperparameter search directory."""
    search_path = Path(search_dir)

    if not search_path.exists():
        print(f"Error: {search_dir} does not exist")
        return

    # Get search name from directory
    search_name = search_path.name

    print(f"Uploading metrics from: {search_dir}")
    print(f"W&B project: {project}")
    print()

    # Find all run directories
    run_dirs = sorted(search_path.glob("run_*"))

    if not run_dirs:
        print("No run directories found")
        return

    print(f"Found {len(run_dirs)} runs")
    print()

    successful = 0
    for run_dir in run_dirs:
        if upload_run_metrics(run_dir, project, entity, search_name):
            successful += 1

    print()
    print(f"Uploaded {successful}/{len(run_dirs)} runs successfully")

    # Also upload summary if exists
    results_file = search_path / "results.json"
    if results_file.exists():
        print()
        print("Creating summary table...")

        with open(results_file) as f:
            results = json.load(f)

        # Create a summary run with table
        summary_id = get_run_id(str(search_path), "summary")
        run = wandb.init(
            project=project,
            entity=entity,
            name=f"{search_name}_summary",
            id=summary_id,
            resume="allow",
        )

        # Create results table
        columns = ["run_id", "learning_rate", "batch_size", "grad_accum",
                   "weight_decay", "warmup_ratio", "val_loss"]
        table = wandb.Table(columns=columns)

        for r in results:
            if r.get("val_loss") is not None:
                c = r["config"]
                table.add_data(
                    r["run_id"],
                    c["learning_rate"],
                    c["batch_size"],
                    c["gradient_accumulation"],
                    c["weight_decay"],
                    c["warmup_ratio"],
                    r["val_loss"],
                )

        wandb.log({"results_table": table})

        # Find best config
        valid_results = [r for r in results if r.get("val_loss") is not None]
        if valid_results:
            best = min(valid_results, key=lambda x: x["val_loss"])
            run.summary["best_val_loss"] = best["val_loss"]
            run.summary["best_config"] = best["config"]

        wandb.finish()
        print("Summary table uploaded")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload hparam search metrics to W&B")
    parser.add_argument("search_dir", type=str,
                        help="Path to hparam search directory")
    parser.add_argument("--project", type=str, default="sft-hparam-search",
                        help="W&B project name")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (username or team)")

    args = parser.parse_args()

    upload_search_results(args.search_dir, args.project, args.entity)


if __name__ == "__main__":
    main()
