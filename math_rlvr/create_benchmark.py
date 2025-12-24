#!/usr/bin/env python3
"""
Create a stratified benchmark subset from the full dataset.
This subset should be representative but small enough for fast parameter sweeps.
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def load_data(jsonl_path: str):
    """Load data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                continue
    return data


def stratified_sample(data, samples_per_level=20, samples_per_subject=15):
    """
    Create a stratified sample ensuring representation across:
    - Difficulty levels (Level 1-5)
    - Math subjects (Algebra, Geometry, etc.)

    Args:
        data: List of data items
        samples_per_level: Target samples per difficulty level
        samples_per_subject: Target samples per subject

    Returns:
        Stratified sample of data
    """
    # Group by level and subject
    by_level = defaultdict(list)
    by_subject = defaultdict(list)

    for item in data:
        level = item.get("level", "Unknown")
        subject = item.get("subject", "Unknown")
        by_level[level].append(item)
        by_subject[subject].append(item)

    print("Data distribution:")
    print(f"\nBy Level:")
    for level, items in sorted(by_level.items()):
        print(f"  {level}: {len(items)} items")

    print(f"\nBy Subject:")
    for subject, items in sorted(by_subject.items()):
        print(f"  {subject}: {len(items)} items")

    # Sample from each level
    sampled = set()

    # Priority 1: Ensure level representation
    for level, items in by_level.items():
        level_samples = min(samples_per_level, len(items))
        selected = random.sample(items, level_samples)
        for item in selected:
            # Use a hash of problem as unique identifier
            sampled.add(json.dumps(item, sort_keys=True))

    # Priority 2: Ensure subject representation (if not already covered)
    for subject, items in by_subject.items():
        # Count how many we already have for this subject
        current_count = sum(
            1 for s in sampled
            if json.loads(s).get("subject") == subject
        )

        if current_count < samples_per_subject:
            # Add more from this subject
            needed = samples_per_subject - current_count
            remaining = [
                item for item in items
                if json.dumps(item, sort_keys=True) not in sampled
            ]
            if remaining:
                to_add = min(needed, len(remaining))
                selected = random.sample(remaining, to_add)
                for item in selected:
                    sampled.add(json.dumps(item, sort_keys=True))

    # Convert back to list
    result = [json.loads(s) for s in sampled]

    return result


def analyze_benchmark(data):
    """Analyze benchmark composition."""
    by_level = defaultdict(int)
    by_subject = defaultdict(int)

    for item in data:
        by_level[item.get("level", "Unknown")] += 1
        by_subject[item.get("subject", "Unknown")] += 1

    print("\nBenchmark composition:")
    print(f"Total samples: {len(data)}")
    print(f"\nBy Level:")
    for level, count in sorted(by_level.items()):
        print(f"  {level}: {count} ({100*count/len(data):.1f}%)")

    print(f"\nBy Subject:")
    for subject, count in sorted(by_subject.items()):
        print(f"  {subject}: {count} ({100*count/len(data):.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create benchmark dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="deepinfra_qwen3_32b_math.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_dataset.jsonl",
        help="Output benchmark JSONL file"
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=20,
        help="Target samples per difficulty level"
    )
    parser.add_argument(
        "--samples-per-subject",
        type=int,
        default=15,
        help="Target samples per subject"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"Loaded {len(data)} total items")

    print(f"\nCreating stratified sample...")
    benchmark = stratified_sample(
        data,
        samples_per_level=args.samples_per_level,
        samples_per_subject=args.samples_per_subject
    )

    analyze_benchmark(benchmark)

    # Write benchmark
    print(f"\nWriting benchmark to {args.output}...")
    with open(args.output, "w") as f:
        for item in benchmark:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Benchmark created with {len(benchmark)} samples")
    print(f"  Estimated training time: ~{len(benchmark) * 0.5 / 60:.1f} minutes for 1 epoch")


if __name__ == "__main__":
    main()
