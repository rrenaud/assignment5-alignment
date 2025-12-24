#!/usr/bin/env python3
"""
Visualize parameter sweep results and Pareto frontier.
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_results(jsonl_file: Path):
    """Load results from JSONL file."""
    results = []
    with open(jsonl_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def print_analysis(results):
    """Print statistical analysis of results."""
    if not results:
        print("No results to analyze")
        return

    # Extract objectives
    thinking_losses = [r["objectives"]["thinking_loss"] for r in results]
    boxed_losses = [r["objectives"]["boxed_loss"] for r in results]
    total_losses = [r["objectives"]["total_loss"] for r in results]

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    print(f"\nThinking Loss:")
    print(f"  Min:  {np.min(thinking_losses):.4f}")
    print(f"  Max:  {np.max(thinking_losses):.4f}")
    print(f"  Mean: {np.mean(thinking_losses):.4f}")
    print(f"  Std:  {np.std(thinking_losses):.4f}")

    print(f"\nBoxed Loss:")
    print(f"  Min:  {np.min(boxed_losses):.4f}")
    print(f"  Max:  {np.max(boxed_losses):.4f}")
    print(f"  Mean: {np.mean(boxed_losses):.4f}")
    print(f"  Std:  {np.std(boxed_losses):.4f}")

    print(f"\nTotal Loss:")
    print(f"  Min:  {np.min(total_losses):.4f}")
    print(f"  Max:  {np.max(total_losses):.4f}")
    print(f"  Mean: {np.mean(total_losses):.4f}")
    print(f"  Std:  {np.std(total_losses):.4f}")

    # Correlation analysis
    print(f"\n\nCorrelation Matrix:")
    print(f"  Thinking vs Boxed:  {np.corrcoef(thinking_losses, boxed_losses)[0,1]:.3f}")
    print(f"  Thinking vs Total:  {np.corrcoef(thinking_losses, total_losses)[0,1]:.3f}")
    print(f"  Boxed vs Total:     {np.corrcoef(boxed_losses, total_losses)[0,1]:.3f}")


def analyze_parameter_impact(results):
    """Analyze impact of each parameter on objectives."""
    print("\n" + "="*70)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*70)

    # Group by each parameter
    params = ["learning_rate", "warmup_ratio", "gradient_accumulation_steps", "max_grad_norm"]

    for param in params:
        print(f"\n{param}:")

        # Group results by parameter value
        by_value = {}
        for r in results:
            value = r["config"][param]
            if value not in by_value:
                by_value[value] = []
            by_value[value].append(r)

        # Compute average objectives for each value
        for value in sorted(by_value.keys()):
            group = by_value[value]
            avg_thinking = np.mean([r["objectives"]["thinking_loss"] for r in group])
            avg_boxed = np.mean([r["objectives"]["boxed_loss"] for r in group])
            avg_total = np.mean([r["objectives"]["total_loss"] for r in group])

            print(f"  {value:>10}: think={avg_thinking:.4f}, boxed={avg_boxed:.4f}, total={avg_total:.4f} (n={len(group)})")


def print_pareto_frontier(pareto_results):
    """Print Pareto frontier details."""
    print("\n" + "="*70)
    print("PARETO FRONTIER ANALYSIS")
    print("="*70)
    print(f"\nFound {len(pareto_results)} Pareto-optimal configurations\n")

    for i, result in enumerate(pareto_results, 1):
        config = result["config"]
        obj = result["objectives"]

        print(f"{i}. Configuration:")
        print(f"   Objectives:")
        print(f"     Thinking: {obj['thinking_loss']:.4f}")
        print(f"     Boxed:    {obj['boxed_loss']:.4f}")
        print(f"     Total:    {obj['total_loss']:.4f}")
        print(f"   Parameters:")
        print(f"     LR:        {config['learning_rate']:.2e}")
        print(f"     Warmup:    {config['warmup_ratio']:.2f}")
        print(f"     GradAccum: {config['gradient_accumulation_steps']}")
        print(f"     GradNorm:  {config['max_grad_norm']:.1f}")
        print()

    # Identify specialized configurations
    print("Specialized Configurations:")
    best_thinking = min(pareto_results, key=lambda r: r["objectives"]["thinking_loss"])
    best_boxed = min(pareto_results, key=lambda r: r["objectives"]["boxed_loss"])
    best_total = min(pareto_results, key=lambda r: r["objectives"]["total_loss"])

    print(f"\n  Best for Thinking Loss: {best_thinking['objectives']['thinking_loss']:.4f}")
    print(f"    LR={best_thinking['config']['learning_rate']:.2e}, "
          f"Warmup={best_thinking['config']['warmup_ratio']:.2f}")

    print(f"\n  Best for Boxed Loss: {best_boxed['objectives']['boxed_loss']:.4f}")
    print(f"    LR={best_boxed['config']['learning_rate']:.2e}, "
          f"Warmup={best_boxed['config']['warmup_ratio']:.2f}")

    print(f"\n  Best for Total Loss: {best_total['objectives']['total_loss']:.4f}")
    print(f"    LR={best_total['config']['learning_rate']:.2e}, "
          f"Warmup={best_total['config']['warmup_ratio']:.2f}")


def create_visualization(results, pareto_results, output_dir: Path):
    """Create visualizations if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("\n⚠ Matplotlib not available. Skipping visualizations.")
        print("  Install with: pip install matplotlib")
        return

    # Extract data
    thinking = [r["objectives"]["thinking_loss"] for r in results]
    boxed = [r["objectives"]["boxed_loss"] for r in results]
    total = [r["objectives"]["total_loss"] for r in results]

    pareto_thinking = [r["objectives"]["thinking_loss"] for r in pareto_results]
    pareto_boxed = [r["objectives"]["boxed_loss"] for r in pareto_results]
    pareto_total = [r["objectives"]["total_loss"] for r in pareto_results]

    # Create figure with 3D scatter
    fig = plt.figure(figsize=(15, 5))

    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(thinking, boxed, total, c='lightblue', alpha=0.6, label='All configs')
    ax1.scatter(pareto_thinking, pareto_boxed, pareto_total,
                c='red', s=100, alpha=0.8, label='Pareto frontier')
    ax1.set_xlabel('Thinking Loss')
    ax1.set_ylabel('Boxed Loss')
    ax1.set_zlabel('Total Loss')
    ax1.set_title('3D Objective Space')
    ax1.legend()

    # 2D: Thinking vs Boxed
    ax2 = fig.add_subplot(132)
    ax2.scatter(thinking, boxed, c='lightblue', alpha=0.6, label='All configs')
    ax2.scatter(pareto_thinking, pareto_boxed, c='red', s=100, alpha=0.8, label='Pareto frontier')
    ax2.set_xlabel('Thinking Loss')
    ax2.set_ylabel('Boxed Loss')
    ax2.set_title('Thinking vs Boxed Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 2D: Total vs Boxed
    ax3 = fig.add_subplot(133)
    ax3.scatter(total, boxed, c='lightblue', alpha=0.6, label='All configs')
    ax3.scatter(pareto_total, pareto_boxed, c='red', s=100, alpha=0.8, label='Pareto frontier')
    ax3.set_xlabel('Total Loss')
    ax3.set_ylabel('Boxed Loss')
    ax3.set_title('Total vs Boxed Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "pareto_frontier_3d.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved 3D visualization to: {output_file}")

    # Parameter impact plots
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    params = ["learning_rate", "warmup_ratio", "gradient_accumulation_steps", "max_grad_norm"]

    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]

        # Group by parameter value
        by_value = {}
        for r in results:
            value = r["config"][param]
            if value not in by_value:
                by_value[value] = {"thinking": [], "boxed": [], "total": []}
            by_value[value]["thinking"].append(r["objectives"]["thinking_loss"])
            by_value[value]["boxed"].append(r["objectives"]["boxed_loss"])
            by_value[value]["total"].append(r["objectives"]["total_loss"])

        values = sorted(by_value.keys())
        thinking_means = [np.mean(by_value[v]["thinking"]) for v in values]
        boxed_means = [np.mean(by_value[v]["boxed"]) for v in values]
        total_means = [np.mean(by_value[v]["total"]) for v in values]

        x = np.arange(len(values))
        width = 0.25

        ax.bar(x - width, thinking_means, width, label='Thinking', alpha=0.8)
        ax.bar(x, boxed_means, width, label='Boxed', alpha=0.8)
        ax.bar(x + width, total_means, width, label='Total', alpha=0.8)

        ax.set_xlabel(param)
        ax.set_ylabel('Loss')
        ax.set_title(f'Impact of {param}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v}' for v in values], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file2 = output_dir / "parameter_impact.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved parameter impact plot to: {output_file2}")


def main():
    parser = argparse.ArgumentParser(description="Visualize parameter sweep results")
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="./param_sweep_results",
        help="Directory with sweep results"
    )

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    # Load results
    results_file = sweep_dir / "sweep_results.jsonl"
    pareto_file = sweep_dir / "pareto_frontier.jsonl"

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    print(f"Loading results from {sweep_dir}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} results")

    if pareto_file.exists():
        pareto_results = load_results(pareto_file)
        print(f"Loaded {len(pareto_results)} Pareto-optimal results")
    else:
        print("No Pareto frontier file found")
        pareto_results = []

    # Analysis
    print_analysis(results)
    analyze_parameter_impact(results)

    if pareto_results:
        print_pareto_frontier(pareto_results)

    # Visualization
    create_visualization(results, pareto_results, sweep_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
